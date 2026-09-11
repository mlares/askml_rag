# AskML public-beta release runbook

AskML is public, bilingual, and answer-only: citations and retrieval traces
remain in API responses, evaluations, and privacy-safe operational diagnostics;
they are not rendered in the visitor interface.

## Production edge

`ask.mlares.space` must terminate at a **global external Application Load
Balancer** whose serverless NEG targets the Cloud Run service. Do not use an
iframe or a forwarding page. Configure the LB to serve the managed TLS
certificate, route only `ask.mlares.space`, and redirect the Cloud Run
`run.app` hostname to the canonical domain. Attach Cloud Armor before opening
the beta:

- managed OWASP/basic WAF protection;
- per-client throttling at the edge; and
- a host rule that only accepts `ask.mlares.space`.

The application’s 10/minute per-process limit, 4 KiB body cap, 1,000-character
question cap, 25-second request deadline, 20-second provider deadline, and
1,200-token initial output cap are a second layer. They are deliberately not a
substitute for Cloud Armor or a distributed quota. For higher traffic, use a
shared counter (for example, Redis) for the per-IP/day budget.

## Deploy a pinned revision

Use this release sequence for every production revision. It deliberately
separates local validation, image publication, deployment, and public checks;
the smoke test does **not** send a valid question to `/ask`, so it does not
incur a model call.

1. Run the quality gate from the repository root:

   ```bash
   uv run pytest -q
   uv run ruff check .
   git diff --check
   ```

2. Build the release image after the evaluated corpus is present. When
   releasing a narrowly scoped hotfix from the currently active image, use that
   immutable image as the base and copy only the reviewed files into a temporary
   Dockerfile. This preserves the active corpus and avoids accidentally
   including unrelated local worktree changes. Start it locally with a dummy
   `OPENAI_API_KEY`, then check `/health`, `/ready`, `/`, and static assets
   without submitting a question.

3. Give the image a new, immutable release tag and publish it. Record the
   digest emitted by `docker push`; never deploy a floating tag.

   ```bash
   docker tag askml:RELEASE \
     southamerica-east1-docker.pkg.dev/askml-505521/askml-images/askml:RELEASE_TAG
   docker push \
     southamerica-east1-docker.pkg.dev/askml-505521/askml-images/askml:RELEASE_TAG
   ```

4. Replace the image in [`deploy/service.yaml`](../deploy/service.yaml) with
   the exact `@sha256:...` digest, then deploy the manifest:

   ```bash
   gcloud run services replace deploy/service.yaml --region=southamerica-east1
   ```

   The OpenAI key belongs only in Secret Manager and the service account
   receives only secret-accessor on that secret.

5. Retrieve the service URL and run the no-cost smoke test. It validates
   `/health`, `/ready`, the landing page, invalid-request behaviour, and the
   JavaScript/CSS contract that hides the answer placeholder after rendering.

   ```bash
   SERVICE_URL="$(gcloud run services describe askml \
     --region=southamerica-east1 --format='value(status.url)')" \
     bash scripts/smoke_deployment.sh
   ```

6. Verify the selected image, latest ready revision, and traffic allocation:

   ```bash
   gcloud run services describe askml --region=southamerica-east1 \
     --format='yaml(status.url,status.latestReadyRevisionName,status.traffic,spec.template.spec.containers[0].image)'
   ```

   Direct a small percentage of traffic to the revision where the release
   strategy permits it, monitor it, then promote it. Keep the prior revision
   available until the release is accepted.

## Transient generation failures

The generation provider makes one additional attempt after a transient failure.
If generation still fails, the API returns a non-answerable result with
an appropriate localized generation error instead of falsely reporting a lack
of retrieved evidence:

- English: `The answer could not be generated or validated. Please try again.`
- Spanish: `No se pudo generar o validar la respuesta. Por favor, intentá nuevamente.`

This response must not be interpreted as evidence that retrieval found no
documents; it is the safe user-facing fallback once generation cannot complete.
Release validation should exercise the retry and fallback paths with unit tests,
not by attempting to induce provider failures in production.

## Rollback

If production validation fails, route all traffic to the known-good revision:

```bash
gcloud run services update-traffic askml \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=southamerica-east1
```

Then rerun `scripts/smoke_deployment.sh` against the returned service URL and
record the reason for the rollback, revision names, and image digests.

## Monitoring and release gates

Create alerts for Cloud Run 5xx rate, latency, instance count, LB/Cloud Armor
denials, and provider/cost budget. Inspect only aggregate, privacy-safe
metrics—never raw visitor questions. Review the public beta weekly for quality,
security, errors, latency, and cost.

Promotion requires seven consecutive days within the agreed error, latency,
and cost ceilings; no high-severity security finding; and a passing release
evaluation for the exact corpus version. The release record must identify the
corpus/chunk manifest, retriever configuration, prompt version, model, and
evaluation output. The acceptance set must have no unsupported factual answer
and must verify Spanish/English plus first-/third-person behaviour and prompt
injection abstention.

## Privacy, conversion, and discovery

Set `BOOKING_URL` to Marcelo’s real Calendly link. `/book` is intentionally a
stable first-party conversion URL, so the UI and future portfolio CTA remain
valid if Calendly paths change. The user-facing bilingual notice is at
`/privacy`; it describes model-provider processing, no-content logging,
contact/removal, and abuse reporting.

Submit the portfolio and AskML sitemaps to Google Search Console and Bing
Webmaster Tools, validate canonical URLs and JSON-LD, and monitor branded
queries and crawl errors. Measure only consented, privacy-conscious aggregate
events: portfolio visit → AskML open → question submit → `/book` click → booked
call, split by language and referrer. The portfolio source is outside this
repository; add its reciprocal AskML, CV, LinkedIn, GitHub, research, and
Calendly links there before promotion.
