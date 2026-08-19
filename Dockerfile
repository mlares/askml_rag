# syntax=docker/dockerfile:1
#
# Cloud Run image for the evaluated AskML RAG public application. Build from
# the askml_rag repository root, where the generated chunk corpus is present.

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

# The public service is planned-BM25 only. Install the small, locked runtime
# subset instead of the development and semantic-retrieval stack, which pulls
# Torch/CUDA packages that Cloud Run does not use for this service.
COPY requirements-cloudrun.txt ./
RUN uv venv && uv pip install --no-cache --requirement requirements-cloudrun.txt

COPY src ./src

# This generated, reviewed corpus is deliberately the only data artifact in
# the public image. The COPY instruction fails if it was not built locally.
COPY data/processed/chunks/chunks.jsonl ./data/processed/chunks/chunks.jsonl

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run injects PORT at runtime. Use one process initially because the
# application keeps its language-specific retrieval indexes in process memory.
CMD ["/bin/sh", "-c", "uvicorn askml_rag.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]
