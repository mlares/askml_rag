# Local HTTP API

This document explains the first web-facing layer of AskML RAG. It does not
replace the retrieval or generation pipeline; it gives another program (and,
later, a web page) a standard way to call it over HTTP.

## The idea

The command-line program asks one question and prints JSON. An HTTP API does
the same work, but a client sends an HTTP request and receives JSON in return.
Here, `POST /ask` means: send a JSON request body to the `/ask` route to ask
one question.

```text
browser or client
  -> POST /ask with {"question": "...", "language": "es"}
  -> FastAPI validates the input
  -> BM25 retrieves the selected top 7 public full-text chunks in that language
  -> grounded generator calls OpenAI and validates the output
  -> JSON response with answer, citations, and retrieval metadata
```

The API deliberately fixes the retrieval choice to the configuration already
selected by evaluation: BM25 over full-text chunks with `k=7`. A caller cannot
silently change the model or retrieve a larger context through the request.

## Components

- `src/askml_rag/api/app.py` defines the HTTP contracts and application
  factory.
- `AskRequest` accepts one non-blank question, up to 1,000 characters, and a
  `language` of `es` or `en` (Spanish is the default). The language selects
  the bilingual retrieval filter and the requested answer language.
- `AskService` is the application layer. It holds one BM25 index and one
  generator in memory, so they are created once when the server starts rather
  than once per request.
- `AskResponse` extends the existing `GroundedAnswer` with the retriever name,
  retrieval limit, retrieved chunk IDs, and the resolved `query_language`.
- `scripts/serve_api.py` is the local server entry point.
- `tests/test_api.py` makes real in-process HTTP requests with FastAPI's test
  client, while injecting a fake LLM. The test has no OpenAI cost or network
  dependency.

An *application factory* is simply a function (`create_app`) that builds the
FastAPI application. Supplying a fake `AskService` to it lets tests replace
external dependencies; calling it with no argument builds the real local
service from the corpus files and `OPENAI_API_KEY`.

## Step 1: prerequisites

Complete the existing local setup first:

1. Run `uv sync` to install project dependencies, including FastAPI and
   Uvicorn.
2. Build the local full-text corpus at
   `data/processed/chunks/chunks.jsonl` using the corpus instructions in the
   README.
3. Put `OPENAI_API_KEY` in the ignored `.env` file, as described in
   [openai_integration.md](openai_integration.md).

The API key stays on the server. A future browser interface calls this API; it
must never contain the OpenAI key itself. This follows the official OpenAI
[API authentication guidance](https://developers.openai.com/api/reference/overview).

## Step 2: run the server locally

From the repository root, run:

```bash
uv run python scripts/serve_api.py
```

Uvicorn starts a local process at `http://127.0.0.1:8000`. Keep that terminal
running. `127.0.0.1` means only programs on your computer can reach it, which
is the safe default for development.

FastAPI also exposes interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

Use it to inspect the schema or send a development request without writing a
client program.

## Step 3: call `POST /ask`

Open a second terminal and run:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"¿Qué experiencia tiene Marcelo con sistemas de recomendación?","language":"es"}'
```

The response has HTTP status `200` and looks conceptually like this:

```json
{
  "answerable": true,
  "answer": "...",
  "claims": [{"text": "...", "citation_ids": ["skills_chunk_001"]}],
  "citations": [{"chunk_id": "skills_chunk_001", "quote": "..."}],
  "retriever": "bm25",
  "retrieval_limit": 7,
  "query_language": "es",
  "retrieved_chunk_ids": ["..."]
}
```

The exact claims and citations vary by question and model output. Treat the
citations—not only fluent answer text—as the evidence to inspect.

## Step 4: understand expected failures

- A blank question, a question longer than 1,000 characters, malformed JSON,
  or unexpected fields returns `422 Unprocessable Content`. That means the
  request did not meet the API contract, so no retrieval or model call occurs.
- A valid but unsupported question returns `200` with `answerable: false`.
  This is a normal, conservative RAG result—not an HTTP error.
- A missing `OPENAI_API_KEY` or corpus file prevents the server from starting.
  Correct the local setup rather than exposing the error details to a public
  client.
- HTTP `429` means the per-client rate limit was reached; honor the
  `Retry-After` response header before trying again.
- HTTP `504` means the answer-generation deadline elapsed, and HTTP `502`
  means the generation provider was unavailable. Both are infrastructure
  outcomes, not evidence-based abstentions.

## Step 5: test without making API calls

```bash
uv run pytest -q tests/test_api.py
```

The tests check successful request handling, invalid blank input, and rejection
of unexpected input fields. They inject `StaticLLM`, so they never use your
key. Run the full gate before committing:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
```

## Current boundary and next work

This is intentionally a small local deployment boundary. It now includes
request IDs, privacy-safe request logs, rate limiting, and generation timeouts;
see [operations.md](operations.md). Before public deployment, add `GET /health`
and `GET /ready`, persistent/shared rate limiting, and authentication or a
suitable public-access policy. Then place the API behind a deployment platform;
the frontend should call the API, while secrets remain in the server environment.
