# Grounded generation

The generation layer converts retrieved `Chunk` objects into a safe,
structured answer. It deliberately has no real LLM provider yet: the boundary
is exercised with `StaticLLM`, a deterministic fake, before a provider can add
network, cost, and non-determinism concerns.

## Contract

`askml_rag.generation.grounded` defines:

- `LanguageModel`: the minimal provider protocol;
- `GenerationRequest`: the question, versioned prompt, and the exact labelled
  public context supplied to the provider;
- `LLMGenerationResponse`: a provider response containing an answerability
  decision, answer, claims, citations, and limitations;
- `GroundedAnswer`: the client-safe result after local validation.

Each substantive claim names one or more citation chunk IDs. The model is
asked for a short verbatim quote for every citation. If it paraphrases the
quote (a common failure mode with PDF-extracted text), the application replaces
it with a deterministic literal excerpt from that same cited chunk and records
a `citation_warnings` entry. The returned citation is enriched with the source
title, document ID, and URL needed by a future API or user interface.

A model may reuse the same chunk for more than one claim. The validator accepts
that pattern and returns one deduplicated citation for the client; it still
rejects any chunk ID that was not retrieved.

## Conservative failure policy

The generator returns a standard abstention instead of an answer when:

- retrieval returned no public chunks;
- a citation refers to a chunk that was not supplied;
- an answerable response has no claims or citations; or
- a claim has no matching citation record.

This makes a wrong source identifier visible as an evidence failure, not as a
plausible-looking answer. The local checks cannot prove that a natural-language
claim is entailed by its cited source; that is a later generation-evaluation
concern.

## Verify the boundary

```bash
uv run pytest -q tests/test_grounded_generation.py
```

## OpenAI adapter

`OpenAIResponsesLLM` implements `LanguageModel` using the OpenAI Responses API
and `responses.parse`. It sends `OpenAIParsedResponse`, a strict Pydantic
schema where every field is required, and returns the existing
`LLMGenerationResponse` contract. The default model is `gpt-5-mini`; pass
`--model` to the command below to compare another compatible model.

For local development, create an ignored `.env` file in the repository root
with this **exact** variable name (not `API_KEY`):

```text
OPENAI_API_KEY='your-key-goes-here'
```

`ask_openai.py` loads that local file without overwriting an already-exported
environment variable. Do not add the key to a source file, notebook, or browser
application. In production, inject it through the deployment platform or a
secret manager instead.

Alternatively, set the secret only in the shell that will run the command:

```bash
export OPENAI_API_KEY='your-key-goes-here'
 
uv run python scripts/ask_openai.py \
  "What experience does Marcelo have with recommendation systems?"
```

The command retrieves the top seven full-text chunks with BM25, sends only
those public chunks to the model, validates the structured result locally, and
prints a JSON `GroundedAnswer`. Start with one query and inspect the citations.
Each API request has cost, so do not run this command in a benchmark loop yet.

Test the adapter without network access or a real key:

```bash
uv run pytest -q tests/test_openai_provider.py
```

For the complete setup and a walkthrough of the request path, see
[OpenAI integration](openai_integration.md).
