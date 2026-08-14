# AskML RAG

AskML RAG is an evaluation-first retrieval-augmented generation project over
Marcelo Lares's public scientific publications and professional material. Its
goal is to answer questions from a bounded corpus, expose the exact papers and
chunks used as evidence, and abstain when the corpus does not support an
answer.

The repository currently implements corpus normalization, PDF and Markdown
ingestion, publication metadata generation, chunking, BM25 and semantic
retrieval, hybrid reciprocal-rank fusion, local Qdrant indexing, a
machine-readable retrieval evaluation runner, and a provider-neutral grounded
generation contract. The generation layer validates structured citations against
the exact retrieved context and abstains when evidence is missing or invalid.
An OpenAI Responses API adapter provides real structured generation over the
local corpus. A FastAPI endpoint and same-origin web interface provide a local
end-to-end application with citations, conservative abstention, rate limiting,
request IDs, provider timeouts, and privacy-safe request logs.

## Current application boundary

The local application is intentionally a public-corpus assistant, not a
general web-search chatbot. It serves a browser interface at `GET /` and a
machine-readable endpoint at `POST /ask`. The server runs BM25 retrieval over
the evaluated full-text corpus (`k=7`), calls the grounded generator, and
returns an answer with literal source quotes or an evidence-based abstention.

The public interface tells visitors that the corpus may be incomplete and that
they should verify citations and not submit private, sensitive, or confidential
information. `POST /ask` is rate limited to 10 requests per client per minute;
responses include `X-Request-ID`, and logs exclude request content and secrets.
These are single-process safeguards for the portfolio deployment, not a
substitute for a production privacy policy, distributed rate limiter, or access
control.

## Start here

Requirements:

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Git

From the repository root:

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

The tests use fake embedders and do not need to download a language model.
Running semantic retrieval for the first time downloads the configured
Sentence Transformers model.

## Repository map

```text
askml_rag/
├── data/
│   ├── evaluation/       # Expert-labelled questions
│   ├── manifests/        # Source identity, provenance, visibility, topics
│   ├── raw/              # Local input PDFs, BibTeX, and Markdown (ignored)
│   └── processed/        # Generated documents, chunks, and indexes (ignored)
├── docs/                 # Scope and development documentation
├── notebooks/            # Exploratory analysis; not the production pipeline
├── reports/              # Generated evaluation reports (ignored)
├── scripts/              # Reproducible command-line workflows
├── src/askml_rag/        # Application package
└── tests/                # Deterministic unit and integration tests
```

The repository uses three representations of the corpus:

1. A manifest describes a source and points to its local input file.
2. Ingestion creates one canonical JSON document per source.
3. Chunking creates retrieval-ready JSONL records with stable document and
   chunk identifiers.

Publication metadata follows a parallel path from BibTeX to a normalized
catalog and one compact summary chunk per paper.

## Rebuild the local corpus

The files under `data/raw/` and `data/processed/` are intentionally ignored by
Git. Before rebuilding, place the approved public inputs at the `source_path`
locations declared by the manifests. In particular, publication workflows
expect:

```text
data/raw/publications/export-bibtex.bib
data/raw/publications/arXiv-*.pdf
```

Generate the publication catalog and paper-summary chunks:

```bash
uv run python scripts/generate_publication_metadata.py
```

Ingest the Markdown sources:

```bash
for manifest in data/manifests/{0*.yaml,skills.yaml,website_*.yaml}; do
  uv run python scripts/ingest_markdown.py "$manifest"
done
```

Ingest the publication PDFs:

```bash
for manifest in data/manifests/arXiv-*.yaml; do
  uv run python scripts/ingest_pdf.py "$manifest"
done
```

Create full-text chunks:

```bash
uv run python scripts/chunk_documents.py \
  --chunk-size-words 250 \
  --overlap-words 40
```

This produces:

- `data/processed/publications.jsonl`: normalized publication metadata;
- `data/processed/publication_summary_chunks.jsonl`: title, authors,
  journal, keywords, and abstract represented as retrieval chunks;
- `data/processed/*.json`: canonical source documents;
- `data/processed/chunks/chunks.jsonl`: full-text retrieval corpus;
- `data/processed/chunks/chunking_manifest.json`: chunking configuration and
  source hashes.

## Search the corpus

Search publication metadata and see which paper was selected:

```bash
uv run python scripts/search_publications.py \
  "Who worked with Marcelo on research about cosmic voids?"
```

Search full-text chunks with BM25:

```bash
uv run python scripts/search_bm25.py \
  "future virialized structures" --limit 5
```

For persistent local semantic search, build and query Qdrant:

```bash
uv run python scripts/index_qdrant.py
uv run python scripts/search_qdrant.py \
  "future virialized structures" \
  --document-type publication \
  --limit 5
```

## Run retrieval evaluation

The unified runner compares `bm25`, `semantic`, and `hybrid` retrieval. It can
search `full_text`, `summaries`, `combined`, or use a `two_stage` summary-first
paper selection strategy.

```bash
uv run python scripts/evaluate_retrieval.py \
  --method bm25 \
  --corpus combined \
  --k 3 5 10
```

Reports are written to `reports/retrieval/`. Each JSON report records the
configuration, aggregate metrics, latency, context size, and the ranked chunk
and document IDs for every benchmark question.

Example semantic and hybrid runs:

```bash
uv run python scripts/evaluate_retrieval.py --method semantic --corpus full_text
uv run python scripts/evaluate_retrieval.py --method hybrid --corpus combined
```

Use `--output PATH` to preserve a named experiment and `--help` to inspect all
options.

## Development workflow

For each change:

1. Update or add the relevant manifest, source, benchmark label, or code.
2. Regenerate only the derived artifacts affected by that change.
3. Run focused tests while iterating.
4. Run the complete quality gate before committing:

   ```bash
   uv run pytest -q
   uv run ruff check .
   git diff --check
   ```

5. Run the relevant retrieval configurations and inspect failures per
   question, not only aggregate scores.
6. Commit source code, tests, manifests, benchmark labels, and selected
   documentation intentionally. Do not accidentally commit private or
   generated corpus files.

The complete ordered workflow, evaluation interpretation, and roadmap are in
[docs/development.md](docs/development.md). The product boundary is defined in
[docs/project_scope.md](docs/project_scope.md).

The grounded-generation contract and its validation behavior are described in
[docs/generation.md](docs/generation.md). The local OpenAI setup, request
flow, security boundaries, and verification commands are in
[docs/openai_integration.md](docs/openai_integration.md). The FastAPI endpoint
and local HTTP workflow are in [docs/api.md](docs/api.md). The built-in web
interface and its architecture are described in [docs/web_ui.md](docs/web_ui.md).
Operational safeguards and their deployment limits are described in
[docs/operations.md](docs/operations.md). The OpenAI model-usage and
cost-estimation probe is documented in
[docs/openai_integration.md](docs/openai_integration.md).
