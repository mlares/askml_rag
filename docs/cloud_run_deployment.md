# Deploy AskML RAG to Google Cloud Run

This procedure publishes the existing local FastAPI application as one public
Cloud Run service. The service provides both the browser UI at `/` and the
same-origin `POST /ask` endpoint. The personal website links to that service;
it does not need to host the application or hold the OpenAI key.

## Scope and preconditions

The deployment image contains application code, static assets, Python runtime
dependencies, and the evaluated retrieval corpus:

```text
data/processed/chunks/chunks.jsonl
```

The corpus is generated and ignored by Git. It is about 9 MB in the current
workspace, so the Docker build must be run from a local checkout where it has
already been built. Do not copy raw PDFs, `data/raw/`, `.env`, reports, private
sources, or other working-tree artifacts into the image.

Before building, confirm that publicly serving the derived chunk text is
compatible with the licences and visibility rules of every indexed source.
The original sources remain authoritative.

The Dockerfile starts the FastAPI application factory with Uvicorn on
`0.0.0.0:$PORT`. Cloud Run injects `PORT` into the container; its default is
`8080` when configured as below.

The public application uses planned BM25, not semantic retrieval or Qdrant.
`requirements-cloudrun.txt` therefore contains only the runtime imports used
by this web service. Its direct dependency versions mirror `uv.lock`; update
the two together whenever one of those runtime imports changes. This avoids
putting Torch/CUDA, notebook, and local experimentation dependencies in the
public container image.

## 1. Validate the release candidate

Run from the repository root:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
test -s data/processed/chunks/chunks.jsonl
```

Use an immutable image tag, normally the commit SHA, so that the deployed
corpus and code revision can be identified together:

```bash
export IMAGE_TAG="$(git rev-parse --short HEAD)"
```

## 2. Build and test the image locally

```bash
docker build --tag askml-rag:local .

docker run --rm \
  --publish 8080:8080 \
  --env PORT=8080 \
  --env OPENAI_API_KEY \
  askml-rag:local
```

In a second terminal, test the UI and one supported request:

```bash
curl --fail http://localhost:8080/

curl -X POST http://localhost:8080/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What experience does Marcelo have with recommendation systems?","language":"en"}'
```

The second request makes a real OpenAI request and can incur cost. The image
does not currently expose a dedicated `/healthz` route; add one before using an
application-level Cloud Run health check.

## 3. Configure the Google Cloud project

Install the Google Cloud CLI, authenticate, and define deployment variables.
Replace the example region with a region appropriate for the intended audience.

```bash
gcloud auth login

export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
export REPOSITORY="askml-rag"
export SERVICE="askml-rag"
export SERVICE_ACCOUNT="askml-rag-run"

gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

Create a private Artifact Registry repository for Docker images:

```bash
gcloud artifacts repositories create "$REPOSITORY" \
  --repository-format=docker \
  --location="$REGION"
```

Create a dedicated Cloud Run service account. It needs only access to the
OpenAI key secret; it does not need broad project access.

```bash
gcloud iam service-accounts create "$SERVICE_ACCOUNT" \
  --display-name="AskML RAG Cloud Run service"

export SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
```

## 4. Store the OpenAI key in Secret Manager

Use a separate OpenAI project key for the public application. Do not reuse a
development key and do not put the value in Git, Docker build arguments, or
the image.

```bash
gcloud secrets create askml-openai-api-key \
  --replication-policy=automatic

printf '%s' "$OPENAI_API_KEY" | \
  gcloud secrets versions add askml-openai-api-key --data-file=-

gcloud secrets add-iam-policy-binding askml-openai-api-key \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

Set an OpenAI project budget alert and restrictive model/rate limits before
making the service public. Budget alerts alone should not be treated as a hard
spending ceiling; preserve the application rate limit and add an edge-level
rate limit before advertising the site broadly.

## 5. Push the container image

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE}:${IMAGE_TAG}"

docker tag askml-rag:local "$IMAGE"
docker push "$IMAGE"
```

## 6. Deploy the public Cloud Run service

The initial configuration limits cost and concurrency. It uses one instance
and one concurrent request because the current rate limiter and retrieval
indexes are in process memory.

```bash
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$SERVICE_ACCOUNT_EMAIL" \
  --set-secrets "OPENAI_API_KEY=askml-openai-api-key:latest" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 1Gi \
  --concurrency 1 \
  --max-instances 1 \
  --min-instances 0 \
  --timeout 30
```

Cloud Run prints the public HTTPS `run.app` service URL after a successful
deployment. `--allow-unauthenticated` is required for visitors to use the
browser UI without Google credentials.

## 7. Verify the deployed revision

```bash
export SERVICE_URL="$(gcloud run services describe "$SERVICE" \
  --region "$REGION" \
  --format='value(status.url)')"

curl --fail "$SERVICE_URL/"

curl -X POST "$SERVICE_URL/ask" \
  -H 'Content-Type: application/json' \
  -d '{"question":"¿Qué experiencia tiene Marcelo con sistemas de recomendación?","language":"es"}'
```

Verify a supported question, an unsupported question, malformed input, and
the `429` rate-limit response. Inspect Cloud Run logs to confirm that logs
contain request IDs and status/timing data but no question text, secrets, or
source corpus content.

## 8. Link from the personal website

For the first release, add a normal link that opens the public service in a
new tab:

```html
<a href="https://YOUR_SERVICE-...run.app/" target="_blank" rel="noopener">
  Ask my public research and professional corpus
</a>
```

Do not embed the service in an iframe initially. It is easier to preserve the
application's privacy notice, citations, and independent error handling in its
own tab.

For a later stable custom subdomain such as `ask.example.com`, use a global
external Application Load Balancer in front of Cloud Run, or Firebase Hosting.
Cloud Run's direct domain-mapping feature is preview/limited availability and
does not map a service below a path such as `example.com/ask`.

## 9. Operational follow-up

Before increasing `--max-instances` or `--concurrency`, replace the in-memory
per-client rate limiter with a proxy-aware/shared implementation and add
application health/readiness endpoints. Add an edge rate limit or bot-control
layer before public promotion, because every accepted `/ask` request can incur
OpenAI usage.

For each corpus or code change, rebuild chunks locally, build a newly tagged
image, push it, deploy it as a new Cloud Run revision, and repeat the deployed
smoke tests. Do not regenerate the corpus at container startup.
