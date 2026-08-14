"""Grounded, structured answer generation over retrieved public chunks."""

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from askml_rag.models import Chunk, Visibility


ABSTENTION_MESSAGE = (
    "I could not find enough validated evidence in the retrieved public sources "
    "to answer this question."
)
PROMPT_VERSION = "grounded-answer-v1"


class CitationReference(BaseModel):
    """A citation emitted by an LLM before it is enriched for a client."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class GeneratedClaim(BaseModel):
    """One substantive answer claim and the evidence chunks supporting it."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class LLMGenerationResponse(BaseModel):
    """Strict provider-neutral response contract for the generation layer."""

    model_config = ConfigDict(extra="forbid")

    answerable: bool
    answer: str = Field(min_length=1)
    claims: list[GeneratedClaim] = Field(default_factory=list)
    citations: list[CitationReference] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answerability(self) -> "LLMGenerationResponse":
        if not self.answerable and (self.claims or self.citations):
            raise ValueError("An abstention cannot contain claims or citations.")
        return self


class ContextChunk(BaseModel):
    """The exactly-labelled evidence supplied to the language model."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    title: str
    source_url: HttpUrl | None
    text: str


class GenerationRequest(BaseModel):
    """A provider-independent request with immutable retrieved context."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    prompt_version: str
    context: list[ContextChunk] = Field(min_length=1)


class LanguageModel(Protocol):
    """Minimal protocol implemented by real providers and deterministic fakes."""

    model_version: str

    def generate(self, request: GenerationRequest) -> LLMGenerationResponse: ...


class GroundedCitation(CitationReference):
    """A validated citation enriched with the source metadata for a client."""

    document_id: str
    title: str
    source_url: HttpUrl | None


class GroundedAnswer(BaseModel):
    """Safe application response after evidence validation."""

    model_config = ConfigDict(extra="forbid")

    answerable: bool
    answer: str
    claims: list[GeneratedClaim] = Field(default_factory=list)
    citations: list[GroundedCitation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    prompt_version: str
    model_version: str
    validation_errors: list[str] = Field(default_factory=list)
    citation_warnings: list[str] = Field(default_factory=list)


class StaticLLM:
    """Deterministic LLM fake for tests and local contract experiments."""

    def __init__(
        self,
        response: LLMGenerationResponse,
        *,
        model_version: str = "fake-static-v1",
    ) -> None:
        self.response = response
        self.model_version = model_version
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> LLMGenerationResponse:
        self.requests.append(request)
        return self.response


def build_context(chunks: Sequence[Chunk]) -> list[ContextChunk]:
    """Create the only evidence records a model is allowed to cite."""
    context: list[ContextChunk] = []
    seen_chunk_ids: set[str] = set()

    for chunk in chunks:
        if chunk.visibility != Visibility.public:
            continue
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        context.append(
            ContextChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                source_url=chunk.source_url,
                text=chunk.text,
            )
        )
    return context


def build_prompt(question: str, context: Sequence[ContextChunk]) -> str:
    """Build a versioned prompt that confines the answer to retrieved evidence."""
    evidence = "\n\n".join(
        "<source "
        f'chunk_id="{chunk.chunk_id}" '
        f'document_id="{chunk.document_id}" '
        f'title="{chunk.title}">\n{chunk.text}\n</source>'
        for chunk in context
    )
    return f"""You are a grounded assistant over a bounded public corpus.

Answer the question using only the source blocks below. Do not use outside
knowledge. If the sources do not support an answer, return answerable=false.
For every substantive claim, provide one or more citation_ids that name source
chunk IDs. Every citation must include a short verbatim quote from that exact
chunk. Never cite a chunk that was not supplied. List each chunk ID only once
in citations; multiple claims may reuse that citation ID.

Return only a response matching this schema:
{{
  "answerable": boolean,
  "answer": string,
  "claims": [{{"text": string, "citation_ids": [string]}}],
  "citations": [{{"chunk_id": string, "quote": string}}],
  "limitations": [string]
}}

Question: {question}

Retrieved sources:
{evidence}
"""


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _source_excerpt(source_text: str, claim_texts: Sequence[str]) -> str:
    """Select a short literal excerpt from a cited source for client display.

    A structured-output model can obey the citation schema while paraphrasing
    the requested quote, particularly for text extracted from PDFs.  This
    helper never invents wording: it selects the source segment with the most
    lexical overlap with the claims that cite it.
    """
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", source_text)
        if segment.strip()
    ]
    if not segments:
        return source_text.strip()[:480] or "Source excerpt unavailable."

    claim_tokens = {
        token
        for claim_text in claim_texts
        for token in re.findall(r"[a-z0-9]{3,}", claim_text.casefold())
    }

    def segment_score(segment: str) -> tuple[int, int]:
        tokens = set(re.findall(r"[a-z0-9]{3,}", segment.casefold()))
        return (len(tokens & claim_tokens), -len(segment))

    return max(segments, key=segment_score)[:480]


def validate_grounding(
    response: LLMGenerationResponse,
    context: Sequence[ContextChunk],
) -> list[str]:
    """Return all structural and evidence violations in an answerable response."""
    if not response.answerable:
        return []

    errors: list[str] = []
    context_by_id = {chunk.chunk_id: chunk for chunk in context}
    citations_by_id: dict[str, CitationReference] = {}

    if not response.claims:
        errors.append("An answerable response must contain at least one claim.")
    if not response.citations:
        errors.append("An answerable response must contain at least one citation.")

    for citation in response.citations:
        chunk = context_by_id.get(citation.chunk_id)
        if chunk is None:
            errors.append(f"Citation '{citation.chunk_id}' was not retrieved.")
        citations_by_id.setdefault(citation.chunk_id, citation)

    cited_ids = set(citations_by_id)
    claim_citation_ids: set[str] = set()
    for claim in response.claims:
        claim_citation_ids.update(claim.citation_ids)
        unknown_ids = set(claim.citation_ids) - cited_ids
        if unknown_ids:
            errors.append(
                "Claim cites no matching citation record: "
                + ", ".join(sorted(unknown_ids))
            )

    unused_ids = cited_ids - claim_citation_ids
    if unused_ids:
        errors.append(
            "Citations are not attached to a substantive claim: "
            + ", ".join(sorted(unused_ids))
        )

    return errors


class GroundedGenerator:
    """Generate a response and abstain unless every citation validates locally."""

    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    def answer(self, question: str, retrieved_chunks: Sequence[Chunk]) -> GroundedAnswer:
        context = build_context(retrieved_chunks)
        if not context:
            return self._abstain("No public evidence was retrieved for this question.")

        request = GenerationRequest(
            question=question,
            prompt=build_prompt(question, context),
            prompt_version=PROMPT_VERSION,
            context=context,
        )
        response = self.llm.generate(request)

        if not response.answerable:
            limitation = (
                response.limitations[0]
                if response.limitations
                else "The retrieved sources do not provide sufficient evidence."
            )
            return self._abstain(limitation)

        errors = validate_grounding(response, context)
        if errors:
            return self._abstain(
                "The generated answer did not provide valid support in the retrieved context.",
                validation_errors=errors,
            )

        context_by_id = {chunk.chunk_id: chunk for chunk in context}
        citations_by_id: dict[str, CitationReference] = {}
        for citation in response.citations:
            citations_by_id.setdefault(citation.chunk_id, citation)

        claim_texts_by_citation_id: dict[str, list[str]] = {
            chunk_id: [] for chunk_id in citations_by_id
        }
        for claim in response.claims:
            for chunk_id in claim.citation_ids:
                claim_texts_by_citation_id[chunk_id].append(claim.text)

        citations: list[GroundedCitation] = []
        citation_warnings: list[str] = []
        for chunk_id, citation in citations_by_id.items():
            source = context_by_id[chunk_id]
            quote = citation.quote
            if _normalized_text(quote) not in _normalized_text(source.text):
                quote = _source_excerpt(
                    source.text, claim_texts_by_citation_id[chunk_id]
                )
                citation_warnings.append(
                    f"Replaced the model quote for chunk '{chunk_id}' with a literal source excerpt."
                )
            citations.append(
                GroundedCitation(
                    chunk_id=chunk_id,
                    quote=quote,
                    document_id=source.document_id,
                    title=source.title,
                    source_url=source.source_url,
                )
            )
        return GroundedAnswer(
            answerable=True,
            answer=response.answer,
            claims=response.claims,
            citations=citations,
            limitations=response.limitations,
            prompt_version=PROMPT_VERSION,
            model_version=self.llm.model_version,
            citation_warnings=citation_warnings,
        )

    def _abstain(
        self,
        limitation: str,
        *,
        validation_errors: Sequence[str] = (),
    ) -> GroundedAnswer:
        return GroundedAnswer(
            answerable=False,
            answer=ABSTENTION_MESSAGE,
            limitations=[limitation],
            prompt_version=PROMPT_VERSION,
            model_version=self.llm.model_version,
            validation_errors=list(validation_errors),
        )
