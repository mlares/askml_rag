# OpenAI API integration

This project uses the OpenAI Responses API only for generation. Retrieval stays
local and deterministic: BM25 selects chunks from the public corpus before the
model is called. The model receives the question and those labelled chunks, not
the entire local dataset.

The adapter uses the Python SDK's `responses.parse` helper with a strict
Pydantic schema. This is the appropriate Structured Outputs pattern for a
model response that the application must render as typed answer, claim, and
citation data. See the official OpenAI [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## What was added

- `openai` is a project dependency in `pyproject.toml`.
- `src/askml_rag/generation/openai_provider.py` implements
  `OpenAIResponsesLLM`, the production adapter behind the provider-neutral
  `LanguageModel` protocol.
- `OpenAIParsedResponse` is the strict schema sent to `responses.parse`. It
  requires `answerable`, `answer`, `claims`, `citations`, and `limitations`.
- `scripts/ask_openai.py` is the command-line entry point.
- `src/askml_rag/config.py` supplies a minimal local `.env` loader.
- `tests/test_openai_provider.py` verifies the adapter with a fake client, so
  tests do not need a network call, API key, or paid request.

## Local setup

1. Install the locked project dependencies from the repository root:

   ```bash
   uv sync
   ```

2. Create an OpenAI API key in the OpenAI platform, then store it locally in
   an ignored `.env` file at the repository root:

   ```text
   OPENAI_API_KEY='your-key-goes-here'
   ```

   The exact variable name matters. `API_KEY` will fail with a clear error;
   the OpenAI SDK and this project use `OPENAI_API_KEY`.

3. Confirm that the key remains local:

   ```bash
   git check-ignore .env
   ```

   The expected output is `.env`. Never commit the key, put it in a notebook,
   or expose it from browser code. For a deployed application, configure the
   same environment variable through the hosting platform's secret manager.
   OpenAI's [API reference](https://developers.openai.com/api/reference/overview)
   recommends loading API keys from a server-side environment variable.

An alternative for one terminal session is:

```bash
export OPENAI_API_KEY='your-key-goes-here'
```

The local loader intentionally does not overwrite an already-exported
environment variable. This lets a deployment or shell-provided secret take
precedence over `.env`.

## Verify before making a real request

Run the adapter tests first:

```bash
uv run pytest -q tests/test_openai_provider.py
```

They use a fake OpenAI client and make no network request. Then make one
interactive request:

```bash
uv run python scripts/ask_openai.py \
  "What experience does Marcelo have with recommendation systems?"
```

The configured default model is `gpt-5.6-luna`. Override it deliberately when
comparing models:

```bash
uv run python scripts/ask_openai.py \
  --model gpt-5.6-luna \
  "What research has Marcelo done on cosmic voids?"
```

Each real request may incur API usage charges. Start with individual questions;
do not use this command as a benchmark loop.

## Measure model usage and estimated cost

`scripts/cost_openai.py` is a deliberate, small model-comparison probe. It is
useful when choosing a model for this project: it sends the same prompt to each
selected model, records returned input/output/reasoning token counts and
latency, then estimates the per-request cost from its local `PRICE_CATALOG`.

It is **not** an account-billing or historical-usage report. The official
OpenAI usage dashboard is the source of truth for organization/project spend;
the official OpenAI documentation also describes using Usage and Costs APIs for
custom reporting. The script's prices are a dated local snapshot and may be
wrong after a pricing change. See the official [usage and cost API guide](https://developers.openai.com/cookbook/examples/completions_usage_api).

First list models visible to the current API project. This makes an API request
but does not generate a model response:

```bash
uv run python scripts/cost_openai.py --list-models
```

To make one low-cost comparison request, name one model explicitly and write
the generated reports under the ignored `reports/` directory:

```bash
uv run python scripts/cost_openai.py \
  --models gpt-5.6-luna \
  --max-output-tokens 128 \
  --csv reports/openai_model_costs.csv \
  --json reports/openai_model_costs.json
```

The default command tests four affordable models; `--all-priced` tests every
priced model visible to the project. Both send real generation requests and can
cost money. Start with the one-model command above and inspect its terminal,
CSV, and JSON output before widening the experiment.

The script loads `OPENAI_API_KEY` from `.env` by default, or accepts another
file with `--env-file PATH`. Its JSON output includes the probe prompt and each
model response, so keep the prompt non-sensitive and do not commit the output
outside an ignored directory.

## Request path and safety checks

```text
question
  -> BM25 retrieves the top 7 full-text chunks locally
  -> GroundedGenerator filters to public chunks and creates labelled context
  -> OpenAIResponsesLLM calls client.responses.parse(...)
  -> Pydantic parses the structured model output
  -> local validator checks citation IDs and claim-to-citation links
  -> JSON GroundedAnswer with citations, abstention, or validation details
```

`OpenAIParsedResponse` controls the shape of the response; the local validator
is still necessary because a correctly shaped JSON response does not establish
that a claim is supported by the retrieved evidence.

The generator abstains if no public evidence is retrieved, the model abstains,
or citations/claims fail the local grounding checks. A citation may be reused
by multiple claims. If a model paraphrases a requested verbatim quote, the
application replaces it with a literal excerpt from the already-cited chunk and
adds a `citation_warnings` entry; it does not silently present model-invented
quote text as evidence.

## Read the output

The command prints one `GroundedAnswer` JSON object:

- `answerable`: whether the system found validated support;
- `answer`: user-facing response, or the standard conservative abstention;
- `claims`: answer statements and the chunk IDs that support them;
- `citations`: literal source excerpts enriched with title, document ID, and
  URL when available;
- `limitations`: why the system abstained or what evidence is unavailable;
- `validation_errors`: grounding failures that caused an abstention;
- `citation_warnings`: transparent repairs to a model-provided quote;
- `prompt_version` and `model_version`: reproducibility metadata.

When inspecting a successful answer, check that its citations answer the
question and that their URLs or titles lead to appropriate public material.
When it abstains, distinguish a retrieval failure (irrelevant chunks) from a
generation validation failure (`validation_errors`) before changing the
retriever or prompt.

## Troubleshooting

| Symptom | Meaning and action |
| --- | --- |
| `OPENAI_API_KEY is not set` | Add the key to `.env` using the exact variable name, or export it in the current shell. |
| `Your .env file uses API_KEY` | Rename the variable to `OPENAI_API_KEY`. |
| `OpenAI returned no parsed structured output` | Inspect the API response/refusal and model compatibility before retrying. |
| `answerable: false` with `validation_errors` | The generated citations did not pass local grounding checks; inspect the cited IDs and retrieved context. |
| `answerable: false` with no validation errors | The model or the retriever found insufficient public evidence; this is an expected conservative outcome. |

Run the full quality gate after changing the adapter, schema, prompt, or
validation policy:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
```
