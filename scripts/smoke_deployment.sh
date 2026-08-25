#!/usr/bin/env bash
# Run after the load balancer is updated, before moving production traffic.
set -euo pipefail

: "${SERVICE_URL:?Set SERVICE_URL to the canonical https://ask.mlares.space URL}"

response_file="$(mktemp)"
trap 'rm -f "$response_file"' EXIT

curl --fail --silent --show-error "$SERVICE_URL/health" >/dev/null
curl --fail --silent --show-error "$SERVICE_URL/ready" >/dev/null
curl --fail --silent --show-error "$SERVICE_URL/" >"$response_file"
grep -q 'AskML · Marcelo Lares' "$response_file"
curl --fail --silent --show-error "$SERVICE_URL/static/app.js" >"$response_file"
grep -q 'emptyState.hidden = true' "$response_file"
curl --fail --silent --show-error "$SERVICE_URL/static/styles.css" >"$response_file"
grep -q '\.empty-state\[hidden\] { display:none; }' "$response_file"

status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --header 'Content-Type: application/json' \
  --data '{"question":" "}' "$SERVICE_URL/ask")"
test "$status" = "422"

echo "AskML smoke test passed: $SERVICE_URL"
