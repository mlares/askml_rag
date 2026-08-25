# AskML RAG improvement opportunities

Audit date: 2026-08-19

This document records improvement opportunities observed in the current local
repository and generated corpus. It is a prioritized engineering backlog, not
a claim that every item is required for a portfolio demonstration.

## Executive assessment

AskML RAG already has a strong foundation: strict Pydantic contracts,
deterministic ingestion and retrieval tests, bilingual provenance, a unified
retrieval evaluator, literal citation checks, conservative abstention, a
same-origin web interface, and a small Cloud Run image. The local quality gate
is healthy.

The largest remaining risks come from the transition from evaluated prototype
to public paid service:

1. cost and abuse controls are process-local while Cloud Run can run multiple
   instances;
2. grounding validation is structural and does not establish that claims are
   entailed by cited text;
3. the current corpus and benchmark expansion is mostly uncommitted and cannot
   be reproduced from a clean clone;
4. the chosen full-text retriever is materially weaker for publication
   discovery and some teaching/profile questions;
5. deployment health, observability, CI, and release provenance are incomplete.

## Audit snapshot

| Area | Current observation |
| --- | --- |
| Quality gate | 98 tests pass; Ruff and `git diff --check` pass. |
| Test warning | One Starlette/FastAPI test-client deprecation warning is emitted. |
| Working tree before this report | 2 modified files and 97 untracked paths, mostly manifests, benchmarks, translations, and corpus inputs. |
| Corpus | 138 documents and 4,241 chunks; about 9 MB of JSONL. |
| Language balance | 2,156 English chunks and 2,085 Spanish chunks. |
| Retrieval baseline | Planned BM25 at `k=7`: chunk recall 0.734, MRR 0.579, nDCG 0.581, document recall 0.836, precision 0.170. |
| Weak retrieval slices | FAMAF: chunk recall 0.574 and MRR 0.354. Scientific publications: document recall 0.352. |
| Context size | The selected retriever supplies about 1,652 words per answerable question on average. |
| Local startup sample | Building the default service took about 2.6 seconds and reached about 227 MB RSS on this machine. |
| Public runtime | Cloud Run deployment is documented; the local application has no `/health` or `/ready` endpoint. |
| Automation | No checked-in CI workflow, coverage gate, type checker, or container smoke-test workflow was found. |

The retrieval values above come from
`reports/retrieval/planned_bm25_full_bilingual_k7_by_topic_language.json`.
That report is currently ignored by Git, so it is evidence in this workspace
but not a reproducible baseline in a clean clone.

## Priority definitions

- **P0 — public-service guardrail:** address before promoting or scaling the
  public endpoint.
- **P1 — next engineering milestone:** correctness, reproducibility, or quality
  work that should precede substantial feature expansion.
- **P2 — hardening:** important for maintainability and production maturity.
- **P3 — optional product evolution:** useful only after the preceding controls
  and measurements exist.

## P0: public-service guardrails

### P0.1 Add a shared usage quota and explicit cost ceilings

Evidence: `InMemoryRateLimiter` keeps counters inside one Python process.
Restarting an instance clears them, and Cloud Run can run two instances. The
OpenAI request does not set `max_output_tokens`. Cloud Run's `max=2` limits
compute scale, not the number or cost of OpenAI requests.

Opportunity:

- enforce per-IP and global daily quotas in a shared atomic store such as
  Firestore, Redis, or an edge gateway;
- retain the short in-process limiter as a first line of defense;
- define an OpenAI output-token ceiling and a maximum request budget;
- add GCP and OpenAI budget alerts, an emergency disable switch, and a
  documented incident procedure;
- decide how trusted proxy headers identify a visitor on Cloud Run, rather
  than assuming `request.client.host` is the original client.

Done when: two application instances share one tested quota, limits survive a
restart, `429` responses include a correct `Retry-After`, and a load test
cannot exceed the configured daily request budget.

### P0.2 Publish an accurate privacy and data-processing notice

Evidence: the interface says that questions are not retained in application
request logs. That is true for local logging, but questions and retrieved
context are still sent to the model provider. The current disclaimer does not
fully explain that processing boundary, retention assumptions, or a contact
route.

Opportunity:

- disclose that the question and retrieved public passages are sent to the
  configured model provider;
- state application-log retention, provider-processing assumptions, cookies,
  analytics, and whether abuse metadata is stored;
- link a short privacy notice from the chatbot and personal website;
- add a contact/removal process for personal or incorrectly indexed material.

Done when: the UI notice and a versioned privacy document describe the same
actual data flow and have been reviewed against the deployed configuration.

### P0.3 Add health, readiness, and deployment verification

Evidence: `/health` and `/ready` are referenced as future work in several
documents but are absent from `create_app`. Deployment verification currently
uses `/`, which only proves that the static page can be returned.

Opportunity:

- add `/health` for process liveness without invoking retrieval or OpenAI;
- add `/ready` that confirms the corpus and retriever loaded successfully;
- configure Cloud Run startup/liveness probes only after those contracts are
  tested;
- add a post-deployment smoke test for `/`, `/ready`, one invalid `/ask`, and a
  controlled fake/staging request;
- document rollback to the previous revision.

Done when: probes have deterministic tests, do not incur provider cost, and a
failed readiness check prevents traffic from reaching an incomplete revision.

### P0.4 Harden the prompt and generation boundary against injection

Evidence: the user question and raw source text are interpolated directly into
an XML-like prompt. Titles and text are not escaped, and the prompt does not
explicitly classify source text as untrusted data. The corpus is curated, but
user questions are public and web/PDF content can contain instruction-like
text.

Opportunity:

- serialize context with an unambiguous escaped representation;
- instruct the model to ignore commands found inside questions and source
  content;
- add adversarial tests for closing tags, instruction injection, fabricated
  citation IDs, hidden instructions in sources, and requests for system data;
- ensure refusals and malformed structured output fail closed without a
  second paid retry loop.

Done when: an adversarial generation suite consistently abstains or returns a
grounded answer and cannot cause non-retrieved content to be cited.

## P1: correctness, answer quality, and reproducibility

### P1.1 Fix disjoint planned-retrieval filters broadening to no filter

Evidence: `PlannedBM25Retriever._merge_filters` intersects caller-supplied and
routed document IDs. A disjoint intersection becomes `[]`, while
`matches_filters` interprets an empty list as “no document restriction.” The
audit confirmed that an unrelated chunk then matches. The public `/ask` route
currently supplies only a language filter, but two-stage or future filtered
callers can hit this path.

Opportunity: represent an explicit match-none result or return no candidates
immediately when two active document scopes are disjoint. Add a regression
test covering that exact case.

Done when: a disjoint filter returns zero results and cannot widen access or
evaluation scope.

### P1.2 Add semantic claim-entailment evaluation

Evidence: grounding currently validates that citation IDs were retrieved and
that claims reference citation records. It does not prove that each claim is
supported by the cited passage. If a model quote is not literal, the system
replaces it with the lexically closest source sentence and still returns the
answer. An answer can therefore be structurally valid but semantically wrong.

Opportunity:

- curate a generation benchmark with supported, partially supported,
  contradictory, unanswerable, and false-premise questions in both languages;
- score claim correctness, citation entailment, completeness, abstention,
  quote fidelity, answer language, and answer concision separately;
- consider failing closed when a requested quote is not literal, or run an
  independently tested entailment check before replacing it;
- validate that every substantive sentence in `answer` maps to a declared
  claim and that declared claims appear in the answer.

Done when: a versioned generation baseline and acceptance thresholds exist,
and a model/prompt change cannot ship solely because retrieval tests pass.

### P1.3 Route publication questions to structured metadata or two-stage retrieval

Evidence: the production API always searches full-text chunks. The project
already has publication summaries and `PublicationCatalogRetriever`, but the
scientific-publication slice has document recall 0.352 at `k=7`; several
metadata questions retrieve none of the expected papers. Counting authors,
papers, dates, or journals from seven passages is also intrinsically unsafe.

Opportunity:

- classify metadata, exhaustive-list, and passage questions;
- answer counts and coauthor lists from the normalized publication catalog;
- use summary-first paper selection followed by full-text evidence for
  paper-specific questions;
- include the selected route and catalog/corpus version in the response and
  evaluation report;
- abstain from exhaustive claims when the catalog cannot prove completeness.

Done when: publication discovery meets a documented threshold on held-out
metadata questions and quantitative answers come from structured records.

### P1.4 Improve weak teaching and professional-profile ranking

Evidence: FAMAF retrieval has chunk recall 0.574 and MRR 0.354. Professional
profile MRR is 0.419. At the same time, mean precision is only 0.170 and the
model receives roughly 1,652 context words.

Opportunity:

- inspect the named failing questions before changing the retriever;
- compare section-aware BM25, multilingual embeddings, hybrid retrieval, and
  a small reranker on the same frozen labels;
- improve Unicode/accent normalization and test Spanish and English lexical
  preprocessing independently;
- add calibrated retrieval confidence and an evidence threshold instead of
  returning every positive BM25 match;
- optimize recall and context precision jointly, not just recall at seven.

Done when: weak slices improve on a held-out set without regressing other
topics or materially increasing latency/context size.

### P1.5 Separate retrieval heuristics from benchmark-specific code

Evidence: `planned_bm25.py` contains hand-authored source prefixes, phrase
cues, boilerplate terms, query expansions, and one hard-coded decomposition.
Several strings mirror benchmark wording. This is inspectable, but it risks
overfitting and becomes difficult to maintain as the corpus grows.

Opportunity:

- move routing/expansion rules into a versioned configuration with documented
  ownership and tests;
- evaluate every rule through ablation and a blind holdout set;
- prefer metadata-driven routing over source-name prefixes;
- record the planner version/rule hash in evaluation and API responses;
- remove rules that do not improve a predeclared metric.

Done when: planner changes are measurable, reviewable data changes and holdout
results determine whether they ship.

### P1.6 Reconcile and version the current corpus expansion

Evidence: the workspace has 97 untracked paths. Many English/Spanish
manifests and FAMAF/IThreex benchmarks exist locally but are not in the current
commit. A clean clone therefore sees a different project state from the
deployed/local image.

Opportunity:

- classify each untracked file as source-of-truth, reviewed benchmark, draft,
  generated artifact, private input, or disposable output;
- commit reviewed manifests, benchmark labels, and compact corpus identity
  indexes together;
- keep drafts clearly named and excluded from official evaluation;
- document which corpus revision produced each image and benchmark report;
- avoid bundling unrelated translation batches and release changes in one
  commit.

Done when: `git status` is intentional, a clean clone contains every reviewed
manifest/label, and the deployed image maps to a code commit plus corpus hash.

### P1.7 Establish one canonical corpus composition

Evidence: `data/processed/chunks/chunks.jsonl` already contains the FAMAF
documents, while `docs/reproduce_retrieval.md` also appends
`famaf_teaching_chunks/chunks.jsonl`. The two files share 101 chunk IDs, so the
documented evaluator command now fails its duplicate-ID guard.

Opportunity:

- choose either one consolidated corpus or explicitly separate shard files;
- generate corpus composition from a versioned manifest rather than shell
  lists;
- reject duplicate document IDs before chunking and duplicate chunk IDs before
  image creation;
- update all README/docs commands to use the same composition.

Done when: the documented clean rebuild and official evaluator command run
without duplicate IDs and produce the expected corpus hash.

### P1.8 Make evaluation reports release-grade artifacts

Evidence: reports record method and counts but not the Git commit, complete
corpus/chunking manifest hash, planner hash, dependency/model revision, or
deployment image digest. All reports are ignored except `.gitkeep`, so the
selected baseline is not available in a clone.

Opportunity:

- include code commit, dirty-tree flag, corpus hash, chunking configuration,
  benchmark hash, retriever/planner version, and embedding model revision;
- select and commit compact baseline reports or publish immutable artifacts;
- add regression thresholds by topic and language, including maximum context
  size and latency;
- distinguish development runs from accepted release baselines.

Done when: a baseline can be reproduced and compared automatically without
relying on unnamed files in one workstation.

### P1.9 Add CI and release automation

Evidence: no CI workflow is checked in. The local gate is good, but untracked
benchmarks and generated corpus are invisible to a clean-clone CI run.

Opportunity:

- run tests, Ruff, `git diff --check`, benchmark schema checks, and package
  build in CI;
- add a small checked-in fixture corpus for end-to-end ingestion/retrieval;
- build and smoke-test the Docker image without a real OpenAI call;
- generate immutable image tags from the commit SHA and record the image
  digest/SBOM;
- require the quality gate before deployment and add a post-deploy verification
  job.

Done when: a fresh runner validates code, schemas, image startup, and release
metadata without private inputs or paid network calls.

## P2: hardening and maintainability

### P2.1 Orchestrate corpus acquisition and rebuilding

Replace shell loops and manual file placement with one idempotent command that
verifies approved source URLs/checksums, validates completeness, ingests,
chunks, builds identity indexes, and writes a signed/hashed build manifest.
Public raw files can live in an immutable artifact bucket or release bundle;
private/unreviewed files must remain outside the public pipeline.

### P2.2 Preserve page, section, and translation provenance in citations

PDF extraction initially knows page numbers, but `CanonicalDocument`, `Chunk`,
`ContextChunk`, and `GroundedCitation` discard them. The generation boundary
also drops `language` and `translation_of`. Add page/section/character offsets,
source-language identity, translation status, and deep-link metadata so users
can verify evidence against the original authority.

### P2.3 Move from word windows to structure-aware stable chunks

Fixed 250-word windows can split headings, tables, list items, and claims.
Index-based chunk IDs also change after earlier text is inserted, forcing
benchmark relabeling. Evaluate heading/page-aware segmentation, repeated
header/footer removal, table handling, and stable anchor/content-derived IDs.
Record both content and metadata hashes so title/topic/source changes invalidate
the correct artifacts.

### P2.4 Consolidate translation workflows and review state

Several translation scripts duplicate batch preparation, cost estimation,
collection, and repair logic. Extract one tested framework and represent
translation model, prompt version, source hash, reviewer, review status, and
completeness checks in metadata. A machine translation should not become
public solely because a provider job succeeded.

### P2.5 Externalize runtime configuration

Model (`gpt-5.6-luna`), corpus path, retrieval limit, timeouts, and rate limits
are constants or constructor defaults. Introduce one validated settings object
fed by environment variables, with safe defaults and startup reporting that
never prints secrets. Resolve paths independently of the current working
directory and derive the API/package version from one source.

### P2.6 Improve provider observability without logging content

Capture provider request ID, input/output token counts, model version,
retrieval/generation latency, answerable/abstention status, timeout class, and
estimated cost as structured metrics. Do not log questions, prompts, quotes,
headers, or keys. Add dashboards and alerts for error rate, p95 latency,
timeouts, spend, abstention drift, and cold starts.

### P2.7 Make timeout and cancellation behavior explicit

`asyncio.wait_for(asyncio.to_thread(...))` returns a timeout to the client but
cannot terminate the worker thread. Keep the provider timeout below the API
deadline, test client disconnects/concurrency, bound the executor, and verify
that timed-out work cannot accumulate or continue generating avoidable cost.

### P2.8 Add production HTTP hardening

Add a restrictive Content Security Policy, `X-Content-Type-Options`, referrer
policy, permissions policy, and appropriate HSTS at the deployment edge.
Decide whether `/docs` and `/openapi.json` should remain public in production.
Keep same-origin requests so broad CORS is unnecessary. Add dependency/image
vulnerability scanning and a secret scan in CI.

### P2.9 Improve UI error handling and accessibility testing

The browser currently converts every non-2xx response into a generic status
message. Render localized handling for `429` plus `Retry-After`, `502`, `504`,
offline state, and request cancellation. Add browser-level tests for keyboard
navigation, focus movement after submission, screen-reader status updates,
mobile layout, source links, language switching, and double submission.

### P2.10 Add broader engineering quality gates

Add coverage reporting with meaningful module thresholds, a type checker,
property tests for schema/filter invariants, load/concurrency tests, and tests
for documented commands. Resolve the Starlette test-client deprecation warning
through a compatible dependency update after confirming FastAPI support.

### P2.11 Simplify dependency and script maintenance

Split notebook/semantic/API dependencies into optional groups so contributors
and CI do not install the full stack. Generate `requirements-cloudrun.txt`
from the lockfile instead of manually mirroring versions. Mark legacy
evaluation scripts as deprecated or remove them after their results are
captured in the unified runner. Consolidate repeated CLI path/config handling.

### P2.12 Harden persistent semantic retrieval before production use

Qdrant indexing currently deletes and recreates a collection and uses
sequential point IDs. Use stable content-derived IDs, batched incremental
upserts, alias-based atomic index swaps, model/dimension compatibility checks,
and rollback. Cache embeddings by content hash and evaluate a multilingual
embedding model before adopting semantic retrieval for the bilingual corpus.

### P2.13 Measure and optimize cold-start/resource headroom

The local sample used roughly 227 MB RSS before serving traffic, against a
documented Cloud Run allocation of 512 MiB. Benchmark the real container under
four concurrent requests, track peak memory and startup time, and set limits
from measurements. Consider serialized indexes, lazy components, or a smaller
corpus representation only if measurements justify the complexity.

### P2.14 Bring documentation back into one current state

Known drift includes:

- `docs/generation.md` says there is no real provider and names `gpt-5-mini`,
  while the adapter exists and defaults to `gpt-5.6-luna`;
- `docs/api.md` still refers to a future browser interface;
- `docs/development.md` says operational logging is future work although it is
  implemented;
- the reproduction guide composes duplicate FAMAF chunks;
- the README still frames the application primarily as local even though a
  public Cloud Run deployment is documented.

Define a documentation owner/language policy, test commands in CI, and update
status sections as part of every release.

### P2.15 Add project and corpus governance files

No top-level license, contribution guide, security-reporting policy, or corpus
data card was found. Add a code license, clarify third-party source and
translation rights separately, document supported/security-reporting channels,
and publish corpus inclusion/exclusion/review criteria.

## P3: optional product evolution

These are deliberately lower priority until safety and measurement are in
place:

- privacy-preserving response caching for repeated public questions;
- response streaming only if structured grounding can still be validated
  before unsupported text is shown;
- user feedback tied to request IDs without collecting question text by
  default;
- conversation history stored client-side or in an explicitly consented
  persistence layer, with each turn independently grounded;
- a custom chatbot subdomain and CDN/edge controls;
- richer source previews, original-versus-translation badges, and page-level
  citation links;
- an administrator-only corpus/revision dashboard;
- scheduled evaluation against the deployed revision with a strict cost cap.

## Suggested execution sequence

### Milestone 1 — safe public beta

1. P0.1 shared quotas and output-token cap.
2. P0.2 privacy notice.
3. P0.3 health/readiness and deployment smoke tests.
4. P0.4 prompt-injection tests and serialization hardening.
5. P1.1 disjoint-filter correctness fix.

### Milestone 2 — reproducible release

1. P1.6 classify and commit the current corpus/benchmark work.
2. P1.7 canonicalize corpus composition.
3. P1.8 version accepted evaluation artifacts.
4. P1.9 add CI, immutable image tags, and release metadata.
5. P2.1 add the corpus rebuild orchestrator.

### Milestone 3 — answer-quality release

1. P1.2 create the bilingual generation/abstention benchmark.
2. P1.3 route publication metadata questions correctly.
3. P1.4 improve weak retrieval slices and context precision.
4. P1.5 validate planner rules on a holdout set.
5. P2.2 and P2.3 improve citation/chunk provenance.

### Milestone 4 — operational maturity

Implement the remaining P2 observability, security, dependency, UI, semantic
index, performance, documentation, and governance work. Select P3 features
only when usage evidence demonstrates a need.

## Audit limitations

- No paid OpenAI request was made.
- No private source contents or secrets were inspected or printed.
- The Cloud Run service was not mutated, load-tested, or redeployed.
- The Docker image was not rebuilt during this audit.
- Existing retrieval reports were inspected; semantic models were not
  downloaded or rerun.
- Local startup time and memory are directional measurements, not Cloud Run
  capacity results.
