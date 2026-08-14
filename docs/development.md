# AskML RAG development guide

This guide describes how to develop AskML RAG from a clean environment to a
grounded question-answering application. Follow the stages in order: each one
creates evidence needed to make the next engineering decision.

## 1. Understand the boundary

AskML RAG answers questions only from an approved, public corpus. It must:

- retrieve relevant passages;
- identify the parent papers or documents;
- support claims with inspectable citations;
- abstain when evidence is insufficient;
- record configurations and results so experiments can be reproduced.

Private, confidential, NDA-protected, and unreviewed personal material must
not enter the public index. The `visibility` field is a runtime safeguard, not
a substitute for reviewing the corpus before ingestion.

Read `docs/project_scope.md` before expanding functionality.

## 2. Prepare the development environment

Clone the repository, enter it, and create the locked environment:

```bash
git clone <repository-url> askml_rag
cd askml_rag
uv sync
```

Verify the environment before touching the corpus:

```bash
uv run pytest -q
uv run ruff check .
```

Useful commands during development:

```bash
uv run pytest -q tests/test_evaluation_runner.py
uv run pytest -q -k publication
uv run ruff check .
uv run ruff check . --fix
```

Review automatic fixes before retaining them. Use `git diff` and `git status`
frequently because local source and generated files may coexist in this
workspace.

### Data policy

The following directories contain local or generated artifacts and are
ignored by Git:

- `data/raw/`
- `data/processed/`
- `reports/`
- `sources/public/` and `sources/private/`

A clean clone therefore needs an explicit, approved data acquisition step.
Until that step is automated, copy the corpus inputs from the controlled local
source and verify that their paths match the manifests. Never infer missing
facts or silently replace a missing source with a different document.

## 3. Add or update a source

Every source needs a YAML manifest in `data/manifests/`. A manifest provides a
stable ID and provenance independently of retrieval results.

Example:

```yaml
document_id: arxiv_1011_5227
title: Properties of satellite galaxies in the SDSS photometric survey
document_type: publication
year: 2011
authors:
  - Marcelo Lares
source_url: https://ui.adsabs.harvard.edu/abs/2011AJ....142...13L
source_path: data/raw/publications/arXiv-1011.5227.pdf
visibility: public
topics:
  - satellite-galaxies
  - astronomy
```

When adding a source:

1. Choose a stable lowercase `document_id` using underscores.
2. Record the original public URL.
3. Set `source_path` to the exact local file.
4. Record a year only when the source provides one.
5. Use a controlled, reusable topic vocabulary.
6. Confirm that `visibility: public` is appropriate.
7. Add or update ingestion tests if the source introduces a new format.

Do not derive benchmark labels from whatever a retriever happens to return.
Labels must be assigned through inspection of the source and generated
chunks.

## 4. Generate publication metadata

The publication catalog is derived from the exported BibTeX file and the
publication manifests:

```bash
uv run python scripts/generate_publication_metadata.py
```

Default inputs and outputs:

```text
Input:  data/raw/publications/export-bibtex.bib
Input:  data/manifests/arXiv-*.yaml
Output: data/processed/publications.jsonl
Output: data/processed/publication_summary_chunks.jsonl
```

Each catalog record contains normalized metadata such as title, authors,
abstract, year, journal, identifiers, keywords, and source links. Each summary
chunk converts that metadata into searchable text while retaining the parent
`document_id`.

Validate the result with both tests and a human-readable query:

```bash
uv run pytest -q tests/test_publication_metadata.py
uv run python scripts/search_publications.py \
  "How many papers did Marcelo publish in ApJ?"
```

The search command retrieves individual papers; aggregation and natural
language answering belong to the later generation layer.

## 5. Ingest canonical documents

The canonical document model gives Markdown and PDF inputs the same fields,
content hash, provenance, and visibility rules.

Ingest one Markdown source:

```bash
uv run python scripts/ingest_markdown.py \
  data/manifests/website_research.yaml
```

Ingest one PDF:

```bash
uv run python scripts/ingest_pdf.py \
  data/manifests/arXiv-1011.5227.yaml
```

Rebuild all currently declared Markdown and publication sources:

```bash
for manifest in data/manifests/{0*.yaml,skills.yaml,website_*.yaml}; do
  uv run python scripts/ingest_markdown.py "$manifest"
done

for manifest in data/manifests/arXiv-*.yaml; do
  uv run python scripts/ingest_pdf.py "$manifest"
done
```

Then verify ingestion behavior:

```bash
uv run pytest -q \
  tests/test_markdown_ingestion.py \
  tests/test_pdf_ingestion.py
```

PDF ingestion assumes text-based PDFs. Scanned documents require an explicit
OCR design and should not be added silently to the existing path.

## 6. Build retrieval chunks

Chunk all top-level canonical JSON documents under `data/processed/`:

```bash
uv run python scripts/chunk_documents.py \
  --chunk-size-words 250 \
  --overlap-words 40
```

The output is:

```text
data/processed/chunks/chunks.jsonl
data/processed/chunks/chunking_manifest.json
```

The manifest records the chunking parameters, source document hashes, and
counts. Treat the chunk size and overlap as experimental parameters. When
either changes:

1. regenerate all chunks;
2. review chunk boundaries and IDs;
3. revise benchmark labels if the evidence boundaries changed;
4. rebuild Qdrant;
5. rerun retrieval evaluation.

Run the chunking tests:

```bash
uv run pytest -q tests/test_chunking.py
```

## 7. Curate the evaluation benchmark

The benchmark lives at `data/evaluation/questions.yaml`. Every answerable
question should define:

- a stable `question_id`;
- the natural-language question;
- category and difficulty;
- an `answer_mode` when the answer comes from publication metadata;
- expected parent `document_id` values;
- relevant full-text `chunk_id` values for passage-mode questions;
- independently reviewed expected claims and their source documents.

`expected_document_ids` are the gold labels for document or paper discovery.
`relevant_chunk_ids` are reserved for full-text passages and must not contain
synthetic publication-summary chunk IDs. Metadata questions use
`answer_mode: metadata` and leave `relevant_chunk_ids` empty. Passage and
unanswerable modes are inferred when `answer_mode` is omitted.

Unanswerable questions must have no expected documents, relevant chunks, or
expected claims. They are needed later to test abstention. Document metrics are
aggregated over all answerable questions; passage metrics are aggregated only
where full-text evidence labels apply.

After editing the benchmark:

```bash
uv run pytest -q tests/test_evaluation_questions.py
```

When canonical documents or chunking parameters change, regenerate the compact
corpus identity index committed with the benchmark:

```bash
uv run python scripts/generate_evaluation_corpus_index.py
```

The index contains IDs, parent-document mappings, and source hashes, but no
source text. This lets benchmark integrity tests run in a clean clone while
also detecting a stale index when the local generated corpus is available.

This catches schema mistakes, unknown fields, duplicate IDs, invalid document
references, and inconsistent answerability labels. It does not replace manual
review of whether a chunk truly supports a claim.

## 8. Establish retrieval baselines

Run the unified evaluator rather than adding another one-off evaluation
script:

```bash
uv run python scripts/evaluate_retrieval.py \
  --method bm25 \
  --corpus full_text \
  --k 3 5 10
```

Supported methods:

- `bm25`: lexical retrieval;
- `semantic`: normalized embedding similarity in memory;
- `hybrid`: reciprocal-rank fusion of BM25 and semantic rankings.

Supported corpus modes:

- `full_text`: search paper and professional-document chunks;
- `summaries`: search one metadata/abstract chunk per publication;
- `combined`: rank full-text and summary chunks together;
- `two_stage`: select papers from summaries, then retrieve full-text evidence
  only from those papers.

Run a useful initial matrix:

```bash
for method in bm25 semantic hybrid; do
  for corpus in full_text summaries combined two_stage; do
    uv run python scripts/evaluate_retrieval.py \
      --method "$method" \
      --corpus "$corpus" \
      --k 3 5 10
  done
done
```

Semantic and hybrid runs load
`sentence-transformers/all-MiniLM-L6-v2` by default. Once it is cached, fully
offline runs can use:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/evaluate_retrieval.py \
  --method hybrid --corpus combined
```

Each report contains:

- the complete experiment configuration;
- per-question expected and retrieved chunk IDs;
- ranked parent document IDs, which identify the retrieved papers;
- Recall, Precision, reciprocal rank, nDCG, and document recall at each `k`;
- retrieved context size;
- per-question latency and aggregate median/p95 latency.

Interpret metrics carefully:

- Chunk recall asks whether all labelled evidence chunks were covered.
- Document recall asks whether the correct parent papers were discovered.
- MRR emphasizes placing the first relevant result early.
- nDCG rewards relevant chunks appearing near the top.
- Precision penalizes irrelevant context that would later distract an LLM.
- Latency and context size expose operational trade-offs.

Inspect low-scoring question records in the JSON report. Decide whether each
failure comes from the corpus, chunk boundaries, benchmark labels, query
wording, or the retriever before changing the implementation.

The older `evaluate_bm25.py`, `evaluate_semantic.py`, and related scripts are
useful historical comparisons. New experiments should go through
`evaluate_retrieval.py` so their configurations and outputs remain comparable.

## 9. Use persistent semantic retrieval when needed

The unified runner currently evaluates in-memory retrieval. Qdrant provides a
comparable persistent local semantic implementation:

```bash
uv run python scripts/index_qdrant.py
uv run python scripts/search_qdrant.py \
  "Who collaborated with Marcelo on cosmic void research?" \
  --document-type publication \
  --limit 5
```

Rebuild the index after changing documents, chunks, the embedding model, or
payload fields. Test both filtering behavior and result parity:

```bash
uv run pytest -q tests/test_qdrant_store.py
```

Qdrant changes storage and filtering characteristics; it should not be
assumed to improve relevance by itself.

## 10. Select and document a retrieval configuration

Before adding an LLM:

1. Freeze a reviewed benchmark version.
2. Run the agreed experiment matrix.
3. Compare answerable questions at `k=3`, `5`, and `10`.
4. Inspect representative successes and failures.
5. Record corpus and chunking configuration with the results.
6. Choose the lowest-cost configuration that satisfies the evidence-coverage
   target, or document why the target is not yet met.
7. Write a human-readable baseline report under `reports/` and intentionally
   decide which report artifacts should be versioned.

Do not optimize only for a single aggregate number. A retrieval configuration
must provide enough relevant evidence without flooding the generation model
with unrelated context.

## 11. Implement grounded generation

Generation is the next application milestone. Implement it only after the
retrieval baseline is understood.

Status: the project now implements items 1--8 with a fake LLM test boundary
and an OpenAI Responses API adapter. The remaining work in this stage is a
small, explicit generation evaluation set. See [generation.md](generation.md)
and [openai_integration.md](openai_integration.md).

Recommended sequence:

1. Define Pydantic output models for the answer, answerability, claims,
   citations, limitations, and model/prompt versions.
2. Define a small LLM protocol and a deterministic fake implementation for
   tests.
3. Build context from retrieved chunks, including chunk ID, title, source URL,
   and text.
4. Add a versioned prompt that prohibits unsupported knowledge and requests
   structured output.
5. Validate that every returned citation was present in the supplied context.
6. Require evidence for each substantive claim.
7. Add conservative abstention for insufficient retrieval and unsupported
   model output.
8. Integrate one real model provider only after unit tests pass.
9. Evaluate generation separately from retrieval so failures can be assigned
   to the correct layer.

Completion gate: answerable questions produce concise cited answers;
unanswerable and false-premise questions abstain; no citation can name a chunk
that was not retrieved.

## 12. Add the API and interface

After generation is stable:

Status: `POST /ask` is implemented as a local FastAPI endpoint over the
selected BM25 full-text configuration at `k=7`. It has request validation and
fake-backed HTTP integration tests. A same-origin web interface is served at
`GET /`; it displays answers, source links and quotes, abstentions, and request
failures. See [api.md](api.md) and [web_ui.md](web_ui.md). Health/readiness
endpoints, authentication, and operational logging remain future work.

1. Add a FastAPI application with dependency-injected retriever and LLM.
2. Implement `POST /api/v1/ask`, `GET /health`, and `GET /ready`.
3. Validate query length, filters, and result limits with Pydantic.
4. Return answers, claims, citations, source metadata, trace ID, timings, and
   configuration versions.
5. Add integration tests with fake retrieval and generation components.
6. Build a minimal interface that shows the answer beside expandable source
   passages and paper metadata.
7. Make abstention and insufficient evidence visible rather than presenting
   them as system errors.

Avoid introducing a complex frontend until the end-to-end behavior is tested.

## 13. Make the project reproducible

The final engineering stage should add:

- one batch ingestion command rather than shell loops;
- deterministic acquisition or documented placement of approved raw data;
- a versioning policy for selected machine-readable reports;
- Docker Compose for the API and optional local services;
- CI running tests, Ruff, and benchmark schema validation;
- structured request logs and latency measurements;
- regression thresholds based on an intentionally selected baseline;
- a public case study describing quality, latency, and resource trade-offs.

## Definition of done for every change

A change is ready when:

- behavior is covered by a focused deterministic test;
- all tests and Ruff pass;
- generated data affected by the change was regenerated;
- benchmark labels were manually checked when chunk IDs changed;
- public/private boundaries were reviewed;
- relevant evaluation configurations were rerun;
- the documentation reflects any changed command, path, schema, or default;
- `git status` contains only intentional files.

Run the final local gate:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
git status --short
```
