# Operational safeguards

The first public-facing safeguards are implemented around `POST /ask`. They
protect the local portfolio application from accidental overuse and make
failures observable without recording sensitive request content.

## Request path

```text
browser
  -> FastAPI middleware assigns a request ID
  -> per-client in-memory rate limit checks POST /ask
  -> validated question reaches AskService
  -> 25-second application deadline bounds generation
  -> OpenAI SDK uses a 20-second network deadline and no automatic retries
  -> response includes X-Request-ID; privacy-safe log event is written
```

The provider deadline is shorter than the application deadline. This leaves a
small interval for the server to translate a provider timeout into a controlled
HTTP response rather than leaving the browser waiting indefinitely.

## Rate limiting

`OperationalSettings` defaults to **10 requests per client per 60 seconds**
for `POST /ask`. A request over the limit returns:

```text
HTTP 429 Too Many Requests
Retry-After: <seconds>
```

Only `/ask` is limited; static assets and the local documentation page remain
available. The implementation uses the connection's client address as a local
key and stores timestamps in process memory.

This is suitable for one local Uvicorn process and a low-traffic portfolio
demo. It is not a distributed production rate limiter: restarting the process
clears the counters, and multiple workers have separate counters. Behind a
reverse proxy, configure trusted proxy handling before relying on a forwarded
client address. For a multi-instance deployment, use a shared store or a
platform/reverse-proxy rate limiter.

## Request IDs and logs

Every response includes an `X-Request-ID` UUID. The server writes one
structured-style log line containing only:

- event name (`http_request`);
- request ID;
- HTTP method and path;
- status code; and
- duration in milliseconds.

It deliberately does **not** log the question, request body, headers, client
address, API key, model prompt, answer, citation text, or stack trace. Provider
and timeout failures add only the request ID and a fixed event name. This makes
a support report actionable without retaining user questions or secrets.

OpenAI recommends request-ID logging for production troubleshooting. This
application's ID is an application-level correlation ID; a future provider
adapter can additionally record the OpenAI request ID without recording the
prompt or secret. See the official OpenAI [API overview](https://developers.openai.com/api/reference/overview).

## Timeouts and provider failures

Two layers bound generation time:

1. `OpenAIResponsesLLM` configures the OpenAI Python client with a 20-second
   timeout and `max_retries=0`. Automatic retries are disabled because retries
   can make a user-visible deadline and request cost unpredictable.
2. The asynchronous `/ask` handler runs the blocking retrieval/generation work
   in a worker thread and applies a 25-second deadline. Exceeding it returns
   `HTTP 504` with a generic message.

Other provider failures return `HTTP 502` with a generic message. The response
does not expose an upstream error body, request payload, or configuration.

Python cannot forcibly stop a blocking network operation already running in a
worker thread. The shorter SDK timeout is therefore the primary mechanism for
ending the upstream request; the API deadline primarily bounds how long the
browser waits.

## Usage and cost probes

The repository includes `scripts/cost_openai.py` for controlled model probes.
It records returned token usage, latency, and a locally estimated per-request
cost after sending real prompts to selected models. It is not a billing-history
report, and its output can contain the prompt and generated text. Keep outputs
under `reports/` and keep probe prompts non-sensitive. Full instructions are in
[openai_integration.md](openai_integration.md#measure-model-usage-and-estimated-cost).

## Public-data disclaimer

The interface displays this boundary before every question:

> This experimental tool uses a curated public corpus and may be incomplete.
> Verify cited sources; do not submit private, sensitive, or confidential
> information.

It tells users what the application is for, makes the corpus limitation
visible, and discourages sensitive input. It is not a substitute for a privacy
policy or legal review when publicly deploying the service.

## Verification

`tests/test_api.py` verifies rate limiting, `Retry-After`, request IDs,
privacy-safe logs, UI disclaimer delivery, and timeout behavior using fake
services. `tests/test_openai_provider.py` verifies that invalid provider
timeout configuration is rejected. None of these tests makes an OpenAI request.

```bash
uv run pytest -q tests/test_api.py tests/test_openai_provider.py
uv run pytest -q
uv run ruff check .
git diff --check
```
