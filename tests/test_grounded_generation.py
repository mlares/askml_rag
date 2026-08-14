from askml_rag.generation.grounded import (
    ABSTENTION_MESSAGE,
    GeneratedClaim,
    GroundedGenerator,
    LLMGenerationResponse,
    StaticLLM,
)
from askml_rag.models import Chunk, Visibility


def make_chunk(*, visibility: Visibility = Visibility.public) -> Chunk:
    return Chunk(
        chunk_id="retrieved_chunk_001",
        document_id="website_projects",
        chunk_index=0,
        document_content_hash="a" * 64,
        title="Selected projects",
        document_type="website",
        source_url="https://www.mlares.space/projects/",
        visibility=visibility,
        text="Marcelo built reproducible data-processing pipelines with validation rules.",
    )


def supported_response() -> LLMGenerationResponse:
    return LLMGenerationResponse(
        answerable=True,
        answer="Marcelo built reproducible data-processing pipelines.",
        claims=[
            {
                "text": "Marcelo built reproducible data-processing pipelines.",
                "citation_ids": ["retrieved_chunk_001"],
            }
        ],
        citations=[
            {
                "chunk_id": "retrieved_chunk_001",
                "quote": "built reproducible data-processing pipelines",
            }
        ],
    )


def test_grounded_generator_returns_enriched_validated_citations() -> None:
    llm = StaticLLM(supported_response())

    answer = GroundedGenerator(llm).answer("What did Marcelo build?", [make_chunk()])

    assert answer.answerable is True
    assert answer.citations[0].chunk_id == "retrieved_chunk_001"
    assert answer.citations[0].document_id == "website_projects"
    assert str(answer.citations[0].source_url) == "https://www.mlares.space/projects/"
    assert 'chunk_id="retrieved_chunk_001"' in llm.requests[0].prompt
    assert "only the source blocks below" in llm.requests[0].prompt


def test_grounded_generator_abstains_without_retrieved_public_evidence() -> None:
    llm = StaticLLM(supported_response())

    answer = GroundedGenerator(llm).answer("What did Marcelo build?", [])

    assert answer.answerable is False
    assert answer.answer == ABSTENTION_MESSAGE
    assert not llm.requests


def test_grounded_generator_excludes_non_public_context() -> None:
    llm = StaticLLM(supported_response())

    answer = GroundedGenerator(llm).answer(
        "What did Marcelo build?", [make_chunk(visibility=Visibility.excluded)]
    )

    assert answer.answerable is False
    assert not llm.requests


def test_grounded_generator_abstains_when_citation_was_not_retrieved() -> None:
    response = supported_response()
    response.citations[0].chunk_id = "invented_chunk_999"
    response.claims[0].citation_ids = ["invented_chunk_999"]

    answer = GroundedGenerator(StaticLLM(response)).answer("What did Marcelo build?", [make_chunk()])

    assert answer.answerable is False
    assert "was not retrieved" in answer.validation_errors[0]


def test_grounded_generator_replaces_a_paraphrased_model_quote_with_source_text() -> None:
    response = supported_response()
    response.citations[0].quote = "Marcelo deployed a Kubernetes cluster"

    answer = GroundedGenerator(StaticLLM(response)).answer("What did Marcelo build?", [make_chunk()])

    assert answer.answerable is True
    assert answer.citations[0].quote == (
        "Marcelo built reproducible data-processing pipelines with validation rules."
    )
    assert answer.citation_warnings == [
        "Replaced the model quote for chunk 'retrieved_chunk_001' with a literal source excerpt."
    ]


def test_grounded_generator_allows_reused_citations_for_multiple_claims() -> None:
    response = supported_response()
    response.claims.append(
        GeneratedClaim(
            text="The pipelines included validation rules.",
            citation_ids=["retrieved_chunk_001"],
        )
    )
    response.citations.append(response.citations[0])

    answer = GroundedGenerator(StaticLLM(response)).answer(
        "What did Marcelo build?", [make_chunk()]
    )

    assert answer.answerable is True
    assert [citation.chunk_id for citation in answer.citations] == [
        "retrieved_chunk_001"
    ]


def test_grounded_generator_preserves_model_abstention() -> None:
    llm = StaticLLM(
        LLMGenerationResponse(
            answerable=False,
            answer="I cannot find the answer.",
            limitations=["The retrieved source does not state that."],
        )
    )

    answer = GroundedGenerator(llm).answer("What did Marcelo build?", [make_chunk()])

    assert answer.answerable is False
    assert answer.answer == ABSTENTION_MESSAGE
    assert answer.limitations == ["The retrieved source does not state that."]
