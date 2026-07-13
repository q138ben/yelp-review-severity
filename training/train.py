"""Vertex CustomJob — fine-tune the ordinal star-rating encoder from gold.training_data.

Reads the leakage-safe train/val/test splits from BigQuery, fine-tunes a sentiment-pretrained
DistilBERT with a single-output regression head (ordinal-as-regression) using gradient clipping,
per-epoch validation monitoring, and early stopping that restores the best epoch's weights, evaluates
on the held-out test split (MAE/RMSE/Spearman/QWK/MacroF1), and writes the model + metrics.json (incl.
the per-epoch training history) to GCS (--model-dir).
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from google.cloud import bigquery, storage
from transformers import AutoTokenizer

# The training procedure lives in src/ordinal.py — the SAME code the notebook runs. The Dockerfile copies
# src/ next to this file (/app/src); locally it's ../src. Add whichever exists to the import path.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE / "src", _HERE.parent / "src"):
    if (_cand / "ordinal.py").exists():
        sys.path.insert(0, str(_cand))
        break
from ordinal import build_model, make_encode, predict, train_ordinal, ordinal_metrics


def load_split(client, table, split, rows=None):
    sql = f"SELECT text, stars FROM `{table}` WHERE split=@s"
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("s", "STRING", split)])
    df = client.query(sql, job_config=cfg).to_dataframe()
    if rows and len(df) > rows:
        df = df.sample(rows, random_state=42).reset_index(drop=True)
    return df


def upload_dir(local_dir, gcs_uri):
    assert gcs_uri.startswith("gs://")
    bucket_name, prefix = gcs_uri[5:].split("/", 1)
    bucket = storage.Client().bucket(bucket_name)
    import os
    for root, _, files in os.walk(local_dir):
        for f in files:
            lp = os.path.join(root, f)
            rel = os.path.relpath(lp, local_dir)
            bucket.blob(f"{prefix}/{rel}").upload_from_filename(lp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--gold-table", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--base-encoder", default="distilbert-base-uncased-finetuned-sst-2-english")
    p.add_argument("--max-seq-len", type=int, default=160)
    p.add_argument("--train-rows", type=int, default=20000)
    p.add_argument("--val-rows", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=100, help="max epochs; early stopping halts at the val plateau")
    p.add_argument("--patience", type=int, default=5, help="stop after this many epochs with no val-QWK gain")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping threshold (0 disables)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    a = p.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bq = bigquery.Client(project=a.project)
    train = load_split(bq, a.gold_table, "train", a.train_rows)
    val = load_split(bq, a.gold_table, "val", a.val_rows)
    test = load_split(bq, a.gold_table, "test")
    if len(val) == 0:  # fallback: carve a val slice from train if the gold split lacks 'val'
        val = train.sample(min(a.val_rows, len(train) // 10), random_state=7)
        train = train.drop(val.index).reset_index(drop=True)
        val = val.reset_index(drop=True)
    print(f"device={dev} | train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base_encoder)
    encode = make_encode(tok, a.max_seq_len)
    model = build_model(a.base_encoder, dev)

    # The fine-tune loop is the shared src/ordinal.train_ordinal — identical to the notebook's run.
    # It returns the best-epoch model explicitly (also mutated in place).
    t0 = time.time()
    model, hist = train_ordinal(model, encode, train, val, dev,
                                epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                                patience=a.patience, max_grad_norm=a.max_grad_norm,
                                log=lambda m: print(m, flush=True))

    metrics = {k: float(v) for k, v in ordinal_metrics(test.stars, predict(model, encode, test.text, dev)).items()}
    metrics["train_rows"] = len(train)
    metrics["val_rows"] = len(val)
    metrics["test_rows"] = len(test)
    metrics["best_epoch"] = hist["best_epoch"]
    metrics["epochs_ran"] = len(hist["history"])
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["history"] = hist["history"]
    print("TEST METRICS:", json.dumps({k: v for k, v in metrics.items() if k != "history"}), flush=True)

    local = "/tmp/model"
    model.save_pretrained(local)
    tok.save_pretrained(local)
    with open(f"{local}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    upload_dir(local, a.model_dir.rstrip("/"))
    print(f"MODEL + metrics written to {a.model_dir}", flush=True)


if __name__ == "__main__":
    main()
