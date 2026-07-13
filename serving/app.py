"""Cloud Run scoring service — POST /predict maps review text to an ordinal star rating,
plus a faithful per-aspect breakdown.

Loads the fine-tuned encoder from GCS (MODEL_URI) at cold start. For each review it returns the overall
1-5 severity score AND an aspect decomposition: the review's sentences are routed to aspects by the shared
lexicon (src/aspects.sentences_by_aspect), and each aspect's sentences are scored by the SAME deployed
encoder — so the "why" is the model's own decomposition of its score, not a separate lexicon signal.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# shared aspect lexicon / sentence router (src/ is copied next to app.py in the image; ../src locally)
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE / "src", _HERE.parent / "src"):
    if (_cand / "aspects.py").exists():
        sys.path.insert(0, str(_cand))
        break
from aspects import aspect_scores

MODEL_URI = os.environ["MODEL_URI"]        # gs://.../models/<run>
LOCAL = "/tmp/model"
MAX_LEN = int(os.environ.get("MAX_SEQ_LEN", "256"))
NEUTRAL = 3.0                              # aspect scores below this are "negative" → flagged as drivers


def download(uri, local):
    bucket_name, prefix = uri[5:].split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    os.makedirs(local, exist_ok=True)
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            continue
        dst = os.path.join(local, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        blob.download_to_filename(dst)


download(MODEL_URI, LOCAL)
tok = AutoTokenizer.from_pretrained(LOCAL)
model = AutoModelForSequenceClassification.from_pretrained(LOCAL)
model.eval()
app = FastAPI(title="Yelp ordinal rating API", version="2.0")


class PredictRequest(BaseModel):
    texts: list[str]


def _score(texts):
    """Continuous 1-5 severity for a list of texts from the deployed encoder (empty list -> empty array)."""
    if not texts:
        return np.array([], dtype=float)
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(-1)
    return np.clip(np.atleast_1d(logits.numpy().reshape(-1)), 1, 5)


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(req: PredictRequest):
    overall = _score(req.texts)
    preds = []
    for i, s in enumerate(overall):
        sc = float(s)
        # Same routing + scorer as the aspect-model notebook: route clauses to aspects, score each aspect's
        # clauses with THIS encoder (shared aspect_scores). Drivers = aspects the encoder reads as negative
        # (below the neutral midpoint), worst first — a SET, not a single argmin, so near-ties (e.g. wait 1.2
        # vs service 1.2) are co-drivers rather than one crowned by noise. Empty when nothing drags it down.
        aspects = {a: round(v, 2) for a, v in aspect_scores(req.texts[i], _score).items()}
        drivers = sorted((a for a, v in aspects.items() if v < NEUTRAL), key=lambda a: aspects[a])
        preds.append({
            "predicted_score": round(sc, 3),
            "predicted_stars": int(np.clip(round(sc), 1, 5)),
            "aspects": aspects,
            "drivers": drivers,
        })
    return {"predictions": preds}
