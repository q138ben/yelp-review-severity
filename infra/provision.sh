#!/usr/bin/env bash
# One-time infra bootstrap for the Yelp ordinal-rating MLOps system (idempotent).
# Region: europe-north1 (Finland) — EU data residency, low-carbon.
# No credential management: the pipeline runs as the yelp-mlops service account (keyless, attached
# identity); local submits use your own gcloud ADC.
#
# PREREQUISITES:
#   1. Create a GCP project and enable billing.
#   2. gcloud auth login && gcloud config set project <PROJECT_ID>   # authenticate + select the project
#   3. gcloud auth application-default login                         # ADC for local submit.py / watch.py
#   (optional) set a budget alert in the console as a guardrail; expected spend is a few dollars.
# Then run this script. Afterwards, see the README "Run it" section (land data, build images, submit pipeline).
set -uo pipefail

PROJECT="${PROJECT:-yelp-review-rating-prediction}"
REGION="${REGION:-europe-north1}"
BQ_LOCATION="${BQ_LOCATION:-europe-north1}"
BUCKET="${BUCKET:-gs://${PROJECT}-mlops}"
AR_REPO="${AR_REPO:-yelp-mlops}"
SA_NAME="${SA_NAME:-yelp-mlops}"
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

echo "== 1. Enable APIs =="
gcloud services enable --project "$PROJECT" \
  storage.googleapis.com \
  bigquery.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  iam.googleapis.com \
  compute.googleapis.com \
  cloudresourcemanager.googleapis.com

echo "== 2. Service account ($SA) — keyless runtime identity, NO key file created =="
gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" \
  --display-name="Yelp MLOps" 2>&1 | tail -1 || echo "  (service account exists)"
# Owner keeps a throwaway demo project simple; for a real project swap in least-privilege roles
# (storage.admin, bigquery.admin, aiplatform.user, artifactregistry.admin, run.admin,
#  iam.serviceAccountUser, cloudbuild.builds.editor).
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA}" --role="roles/owner" --condition=None 2>&1 | tail -1

echo "== 3. GCS bucket ($BUCKET) =="
gcloud storage buckets create "$BUCKET" \
  --project "$PROJECT" --location "$REGION" \
  --uniform-bucket-level-access --public-access-prevention 2>&1 | tail -1 \
  || echo "  (bucket exists)"

echo "== 4. BigQuery datasets (bronze / silver / gold) =="
for ds in bronze silver gold; do
  bq --location="$BQ_LOCATION" mk --dataset --description "medallion: $ds" \
    "${PROJECT}:${ds}" 2>&1 | tail -1 || echo "  ($ds exists)"
done

echo "== 5. Artifact Registry (docker) =="
gcloud artifacts repositories create "$AR_REPO" \
  --project "$PROJECT" --repository-format=docker --location "$REGION" \
  --description="Yelp MLOps images" 2>&1 | tail -1 || echo "  (repo exists)"

echo "== SUMMARY =="
echo "service account:"; gcloud iam service-accounts list --project "$PROJECT" \
  --filter="email:${SA}" --format="value(email)" 2>&1
echo "bucket : $BUCKET"; gcloud storage ls "$BUCKET" 2>&1 | head -1 || true
echo "datasets:"; bq ls --project_id "$PROJECT" 2>&1 | tail -5
echo "AR repo:"; gcloud artifacts repositories list --project "$PROJECT" --location "$REGION" --format="value(name)" 2>&1
echo "PROVISION DONE"
