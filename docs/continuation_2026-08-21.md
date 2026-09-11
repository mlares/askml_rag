# AskML continuation plan — 2026-08-21

This is the practical handoff for resuming the AskML public-beta work.

## Current state

The AskML application has been prepared for a public beta:

- bilingual Spanish/English UI and language-aware answers;
- first-person handling for direct questions and third-person handling for
  questions about Marcelo;
- grounded retrieval with internal-only citation traces (no evidence panel in
  the visitor UI);
- bounded requests: 1,000-character question limit, JSON-only requests, 4 KiB
  body cap, request IDs, local per-IP rate limiting, provider and application
  timeouts, a 1,200-token initial output cap, and one retry with a doubled cap;
- prompt-injection boundary: retrieved text is explicitly untrusted reference
  material and cannot override the answer rules;
- `GET /health`, `GET /ready`, `GET /privacy`, and `GET /book` endpoints;
- a bilingual privacy notice, booking CTA, canonical metadata, Open Graph
  metadata, and Person/WebSite JSON-LD; and
- Cloud Run deployment and smoke-test artifacts.

All local checks passed at the last implementation point:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
```

Result: **110 tests passed** (with one third-party TestClient deprecation
warning).

## Current deployment inputs

`deploy/service.yaml` is now filled with these values:

```text
Project: askml-505521
Service account: askml-runner@askml-505521.iam.gserviceaccount.com
Image: southamerica-east1-docker.pkg.dev/askml-505521/askml-images/askml@sha256:9bc5c36882a94a8209813105065e15ea5ad4383287f86bf7f89063c6aa275f1d
Calendly: https://calendly.com/marcelo-lares/30min
Region: southamerica-east1
Revision: askml-00007-9zn
```

The image was built with Cloud Build, tagged as `v7-rag-retrieval`, pushed to
Artifact Registry, and referenced by its immutable digest. A digest freezes the exact code,
dependencies, static UI, and reviewed corpus included in that build. Do not
rebuild merely to deploy it.

If the Cloud Run command has not yet been run, deploy this exact image with:

```bash
gcloud run services replace deploy/service.yaml --region=southamerica-east1
```

If it has been run, retrieve the service URL and revision for the deployment
record:

```bash
gcloud run services describe askml --region=southamerica-east1 \
  --format='yaml(status.url,status.latestReadyRevisionName,status.traffic)'
```

## Immediate next checks

Use the returned Cloud Run URL first, before configuring the custom domain:

```bash
SERVICE_URL='https://PASTE_THE_RUN_APP_URL'
curl --fail-with-body "$SERVICE_URL/health"
curl --fail-with-body "$SERVICE_URL/ready"
curl --fail-with-body "$SERVICE_URL/"
```

For the supplied smoke script, use the canonical domain only after it exists:

```bash
SERVICE_URL='https://ask.mlares.space' bash scripts/smoke_deployment.sh
```

Then make one deliberate real question in the UI and verify that:

- a Spanish question gets a Spanish response;
- an English question gets an English response;
- direct “you” and third-person “Marcelo/he” questions use the right voice;
- unsupported questions abstain rather than invent facts;
- `/book` redirects to the Calendly URL; and
- `/privacy` displays the bilingual notice.

## Remaining public-beta work

### 1. Production domain and edge security

Create a serverless NEG for Cloud Run and attach it to a global external
Application Load Balancer. Configure managed TLS and DNS for
`ask.mlares.space`. This must be a real reverse-proxy route—remove any iframe
or forwarding-page wrapper.

At the edge, add Cloud Armor with managed WAF rules and per-client rate
limiting. Route only `ask.mlares.space` to the service and redirect/directly
block the raw `run.app` origin as appropriate for the chosen load-balancer
configuration. The local limiter is intentionally a second layer, not a
distributed production quota.

### 2. Observe and protect costs

Create alerts for Cloud Run 5xx rate, latency, instance count, load-balancer/
Cloud Armor denials, and the OpenAI cost budget. Logs must remain aggregate
and privacy-safe: never log raw questions, prompts, answers, or source text.

For traffic beyond the limited beta, add a shared per-IP/day quota (for example
Redis) so that limits apply across Cloud Run instances.

### 3. Portfolio and discovery work

The source of `www.mlares.space` is not in this repository. Update it to add:

- a bilingual AskML link and Calendly CTA on the contact page;
- concise bilingual recruiter landing content: role focus, demonstrated
  outcomes, selected case studies, downloadable CV, and booking CTA;
- reciprocal canonical links to AskML, CV, LinkedIn, GitHub, and research
  identities; and
- canonical URLs, Person/WebSite JSON-LD, Open Graph cards, robots, sitemap,
  and sitemap validation.

Register the portfolio and AskML in Google Search Console and Bing Webmaster
Tools; submit sitemaps, investigate crawl errors, and monitor branded search
for identity collisions.

### 4. Controlled release and iteration

Launch with limited promotion. Review quality, cost, security, and errors
weekly. Keep the prior Cloud Run revision available for rollback:

```bash
gcloud run services update-traffic askml \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=southamerica-east1
```

Promote only after seven consecutive days within the agreed error, latency,
and cost ceilings; no high-severity security issues; and no regressions in the
approved bilingual evaluation set. Record the corpus/chunk manifest, retriever
configuration, prompt version, model, image digest, and evaluation report for
every promoted revision.

## Future code changes: repeatable release loop

Only when code or the corpus changes:

```bash
uv run pytest -q
uv run ruff check .
docker build -t askml:local .

RELEASE_TAG='vNEXT'
IMAGE="southamerica-east1-docker.pkg.dev/askml-505521/askml-images/askml:${RELEASE_TAG}"
docker tag askml:local "$IMAGE"
docker push "$IMAGE"
docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}'
```

Update `deploy/service.yaml` with the returned `@sha256:...` image reference,
deploy it, smoke-test it, and only then shift traffic. Never substitute the
local Docker image ID for the Artifact Registry digest.

## Related documents

- [`public_beta.md`](public_beta.md): production edge, release gates, and
  privacy/discovery overview.
- [`cloud_run_deployment.md`](cloud_run_deployment.md): Cloud Run, Secret
  Manager, and service-account setup detail.
- [`operations.md`](operations.md): application safeguards and privacy-safe
  logging.
- [`deploy/service.yaml`](../deploy/service.yaml): active deployable manifest.
- [`scripts/smoke_deployment.sh`](../scripts/smoke_deployment.sh): no-model-cost
  post-deploy smoke test.
