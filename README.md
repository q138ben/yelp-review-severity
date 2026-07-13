# Yelp review → star-rating: case study + production MLOps system

Predict a review's **1–5 star rating from its free text**, framed as *severity/priority triage* for an
operations stakeholder (a review ≈ a free-text incident report). Two parts:

1. **Analysis & modelling** — reproducible notebooks (EDA → ordinal model → interpretable aspect model).
2. **Production system** — a **YAML-driven Vertex AI Pipeline** that runs a **medallion** data pipeline on
   BigQuery, fine-tunes the ordinal encoder, gates on **QWK**, and deploys a scoring API to **Cloud Run**.

> **Data licence:** the Yelp Open Dataset is *academic-use-only, non-commercial*. Everything here is a
> **private demonstration** — the GCS bucket/BigQuery datasets are private and this is not a commercial
> deployment on Yelp data. In a real deployment you'd swap in your own/licensed data; the architecture is identical.

---

## Presentation

The case-study slide deck (self-contained HTML), viewable via GitHub Pages:

**▶ [View the slide deck](https://q138ben.github.io/yelp-review-severity/docs/presentation.html)**

Navigate with arrow keys / space; `F` for fullscreen. Source: [`docs/presentation.html`](docs/presentation.html).

---

## Notebooks — rendered with charts

GitHub can't display the notebooks' interactive Plotly charts inline, so a rendered HTML copy of each
(charts baked in) lives alongside the `.ipynb` in [`notebooks/`](notebooks/), viewable via GitHub Pages:

- **[01 · EDA](https://q138ben.github.io/yelp-review-severity/notebooks/01_eda.html)** — whole-dataset EDA, data quality, leakage-safe split
- **[02 · Ordinal model](https://q138ben.github.io/yelp-review-severity/notebooks/02_ordinal_model.html)** — training diagnostics + results
- **[03 · Aspect decomposition](https://q138ben.github.io/yelp-review-severity/notebooks/03_aspect_model.html)** — per-aspect breakdown

---

## Repo layout

```
config.yaml                      # single source of truth (project, region, model params, QWK gate)
notebooks/                       # 01 EDA (+ data quality, language) · 02 ordinal model · 03 aspect model
src/                             # shared helpers (eda_utils, ordinal metrics, aspects)
infra/
  provision.sh                   # one-time bootstrap: APIs, service account, bucket, BQ datasets, Artifact Registry
pipelines/
  medallion/01_bronze.sql        # LOAD raw JSON -> BQ bronze
  medallion/02_silver.sql        # clean / type / dedupe / leakage-safe split
  medallion/03_gold.sql          # features + ordinal label + train/val/test
  pipeline.py                    # KFP v2 pipeline definition
  compiled/pipeline.yaml         # compiled artifact (what Vertex runs)
  submit.py                      # submit the pipeline run
training/                        # Vertex CustomJob: train.py + Dockerfile + cloudbuild.yaml
serving/                         # Cloud Run FastAPI: app.py + Dockerfile + cloudbuild.yaml
```

---

## Production architecture

```
        RAW (GCS bronze/)                 BigQuery (medallion)                Vertex AI Pipeline
 ┌───────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────┐
 │ review.json  business.json │   │ bronze → silver → gold        │   │ bq(bronze)→bq(silver)→bq(gold)│
 │  (immutable landing)       │──▶│ clean · type · dedupe · split │──▶│      → train (CustomJob)      │
 └───────────────────────────┘   │ features + ordinal label      │   │      → read metrics           │
                                  └──────────────────────────────┘   │      → [QWK ≥ gate?]          │
                                                                      │            └─▶ deploy Cloud Run│
  Artifact Registry: train:latest, serve:latest                      └──────────────────────────────┘
  Identity: yelp-mlops SA (keyless)                                             │
                                                                                ▼
                                                        Cloud Run: POST /predict  (scale-to-zero)
```

- **Medallion** — *bronze* = raw JSON in GCS; *silver* = cleaned/typed/deduped BigQuery tables with the
  leakage-safe business-level split; *gold* = ML-ready `text → stars` set (+ restaurant context), leakage
  features (user/business avg-star) deliberately excluded.
- **Orchestration** — Vertex AI Pipelines (KFP v2), compiled to `pipelines/compiled/pipeline.yaml`. SQL steps run as
  BigQuery jobs; training runs as a Vertex CustomJob; deploy runs only if the **QWK gate** passes.
- **Serving** — Cloud Run FastAPI (`POST /predict`), image in Artifact Registry, **min-instances 0** (near-zero
  idle cost), loads the model from GCS at cold start.
- **Region** — `europe-north1` (Finland): EU data residency, low-carbon.

---

## Results (live run `yelp-ordinal-rating-20260710231037`)

Medallion + train + eval + deploy ran on Vertex AI. Held-out **test** metrics (n=20,179), model trained on a
20k sample (~2.3 h on an `e2-highmem-8` CPU job):

| MAE ↓ | RMSE ↓ | Spearman ↑ | **QWK ↑** | Acc ↑ |
|---|---|---|---|---|
| 0.432 | 0.677 | 0.828 | **0.879** | 0.660 |

**QWK 0.879 ≥ 0.80 gate → passed.** Deployed to Cloud Run, private (identity-token auth), scale-to-zero:

```
$ curl -H "Authorization: Bearer $(gcloud auth print-identity-token --audiences=$URL)" \
    -d '{"texts":["great food but the wait was over an hour and the server was rude"]}' $URL/predict
{"predictions":[{"predicted_score":1.939,"predicted_stars":2,
  "aspects":{"food":4.92,"wait":1.15,"service":1.27},"drivers":["wait","service"]}]}
```
The `aspects`/`drivers` come from re-scoring each aspect's *clauses* with the **same deployed encoder** (the
lexicon only routes clauses) — a faithful decomposition of the model's own score: it reads the *food* clause 4.9
but *wait*/*service* ~1.2, so those are the drivers. Endpoint: `https://yelp-rating-api-kipxvjlppq-lz.a.run.app`

---

## Environment

Python **3.12.8**, managed with **pipenv** (all dependencies pinned in `Pipfile.lock`). Reproduce the exact
environment before running anything below:

```bash
PIPENV_VENV_IN_PROJECT=1 pipenv sync          # builds ./.venv deterministically from the lockfile
```

Then invoke project commands with `./.venv/bin/python …` (as shown below) or `pipenv run python …`.

---

## Run it

```bash
# 0. one-time GCP setup — prerequisites (create project + billing, gcloud auth) are in provision.sh's header:
bash infra/provision.sh                       # APIs, service account, bucket, datasets, registry

# 1. land raw data in bronze (GCS)
gcloud storage cp "Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json" gs://<bucket>/bronze/business/
gcloud storage cp "Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json"   gs://<bucket>/bronze/review/

# 2. build the images (native amd64 via Cloud Build). Both build from the repo ROOT so they ship src/
#    (train: the shared training core; serve: the shared aspect lexicon); a .gcloudignore keeps uploads tiny.
gcloud builds submit . --config=training/cloudbuild.yaml --region=europe-north1 --service-account=<SA>
gcloud builds submit . --config=serving/cloudbuild.yaml  --region=europe-north1 --service-account=<SA>

# 3. compile + submit (keyless: the pipeline runs as the yelp-mlops SA; you submit with your gcloud ADC)
.venv/bin/python pipelines/pipeline.py                       # -> pipelines/pipeline.yaml
.venv/bin/python pipelines/submit.py                         # full run: medallion → train → eval → deploy
# ...or redeploy an existing model without retraining (deploy-only branch of the same pipeline):
.venv/bin/python pipelines/submit.py \
  --param train_model=False --param model_dir=gs://<bucket>/models/<run>

# 4. call the deployed API (private; use an identity token)
TOKEN=$(gcloud auth print-identity-token)
URL=$(gcloud run services describe yelp-rating-api --region=europe-north1 --format='value(status.url)')
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"texts":["great food but the wait was over an hour"]}' "$URL/predict"
```

---

## Model versioning & rollback

Each training run writes to an **immutable** path `gs://<bucket>/models/<timestamp>` (minted by `submit.py`), and
the deploy step bakes that exact path into the Cloud Run revision's `MODEL_URI` env var. So every train → a
distinct artifact → a new, rollback-able revision (no mutable "latest" that silently changes under running
instances). To **roll back**, redeploy an earlier version without retraining:

```bash
.venv/bin/python pipelines/submit.py --param train_model=False \
  --param model_dir=gs://<bucket>/models/<older-timestamp>
```
Cloud Run also retains prior revisions, so you can shift traffic back instantly with
`gcloud run services update-traffic yelp-rating-api --to-revisions=<rev>=100`.
