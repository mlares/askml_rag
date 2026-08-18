# Bilingual curated dataset

The bilingual corpus is a retrieval improvement, not new evidence. Every
translation is derived from one reviewed source and keeps that relationship in
its manifest:

```yaml
language: en
translation_of: cv_education
```

The original source remains the authority. A machine translation must never be
represented as an independently published source or used to add claims.

## Scope

`scripts/translate_bilingual_dataset.py` translates these non-paper sources:

- public-CV Markdown documents;
- the curated skills document;
- local captures of the personal website pages;
- `personal_traits.md`, only after the owner confirms that its contents are
  appropriate for this public, externally processed corpus.

`scripts/translate_publication_sources.py` separately translates the canonical
text extracted from every publication PDF. It preserves bibliographic titles,
DOIs, arXiv IDs, equations, references, and numeric data. The English PDF
remains authoritative; the Spanish Markdown is a derived retrieval copy.

## Batch workflow and cost boundary

Run from the repository root:

```bash
uv run python scripts/translate_bilingual_dataset.py prepare
```

This creates these local, ignored artifacts:

```text
reports/bilingual_translation/batch_input.jsonl
reports/bilingual_translation/tasks.json
```

`tasks.json` records every source hash, output-token ceiling, and a conservative
cost estimate. Inspect it before sending anything. Use `prepare --only-missing`
to prepare only source counterparts whose target files do not exist.

Submit only when the displayed estimate is at or below the chosen ceiling:

```bash
uv run python scripts/translate_bilingual_dataset.py submit \
  --max-cost-usd 0.50
```

The script refuses to submit an estimated over-budget job. The estimate is a
local upper bound from configured Batch prices and output-token limits, not an
invoice or a guarantee of future provider pricing. Batch processing is
asynchronous; check it later:

```bash
uv run python scripts/translate_bilingual_dataset.py status
```

After it reports `completed`, apply it:

```bash
uv run python scripts/translate_bilingual_dataset.py apply
```

`apply` refuses partial output, fenced text, malformed question YAML, and
results whose input source changed after preparation. It writes translation
sources, translation manifests, and `questions_es.draft.yaml`.

Publication translations use the same guarded lifecycle and a separate set of
artifacts:

```bash
uv run python scripts/translate_publication_sources.py prepare
uv run python scripts/translate_publication_sources.py submit \
  --max-cost-usd 0.65
uv run python scripts/translate_publication_sources.py status
uv run python scripts/translate_publication_sources.py apply
```

The publication workflow must be followed by the completeness test below.
Provider success status alone does not prove that a long translation retained
the whole paper. If any derived text is shorter than 75% of its canonical
source, repair only those papers using bounded, paragraph-preserving segments:

```bash
uv run python scripts/repair_publication_translations.py prepare
uv run python scripts/repair_publication_translations.py submit \
  --max-cost-usd 0.24
uv run python scripts/repair_publication_translations.py status
uv run python scripts/repair_publication_translations.py apply
```

After applying either workflow, ingest the new Markdown manifests and rebuild
the unified chunk corpus. `tests/test_bilingual_manifests.py` verifies unique
document IDs, explicit languages, exactly one opposite-language counterpart
per original, locally available source paths, and full-publication translation
length completeness.

## API-key permissions

The Batch API requires a key that can upload an input file and create/read a
batch. A restricted key missing `api.files.write` will fail before any batch is
created. Create or update a project key with Files read/write and the relevant
Batch/Responses permissions, put that key in the local ignored `.env` as
`OPENAI_TRANSLATE_API_KEY`, and rerun `submit`. This keeps it separate from
the app's `OPENAI_API_KEY`. Never place either key in Git or in the Batch JSONL
file.

## Why translated questions remain a draft

The source-language question labels include exact `relevant_chunk_ids`.
Translated documents can split into different word chunks, so blindly copying
those IDs would create false ground truth. The generated Spanish question file
is intentionally named `questions_es.draft.yaml`.

To turn it into a curated evaluation set:

1. Review translations for factual and Markdown fidelity.
2. Ingest all original and translated manifests.
3. Re-chunk the corpus and regenerate `corpus_index.json`.
4. For each translated answerable question, inspect retrieved evidence and
   label the actual relevant translated chunk IDs.
5. Rename the reviewed file to `questions_es.yaml`, validate it with the
   existing Pydantic schema, and run retrieval evaluation separately for each
   language.

This preserves the project's evaluation-first rule: generated text can be a
candidate dataset artifact, but it is not ground truth until reviewed.
