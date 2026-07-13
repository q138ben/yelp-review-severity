#!/usr/bin/env bash
# Smoke-test the deployed rating API with a JSON request file.
# Usage: ./predict.sh [request.json]   (defaults to sample_request.json)
set -euo pipefail

REGION="${REGION:-europe-north1}"
SERVICE="${SERVICE:-yelp-rating-api}"
REQ="${1:-$(dirname "$0")/sample_request.json}"

URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
TOKEN="$(gcloud auth print-identity-token --audiences="$URL")"

curl -sS --fail-with-body -m 180 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @"$REQ" \
  "$URL/predict"
echo
