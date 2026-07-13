"""Ordinal-regression metrics + the shared fine-tuning core for the Yelp star-rating case study.

This module is the single source of truth for the training procedure: `build_model`, `make_encode`,
`predict`, and `train_ordinal` are imported by BOTH the analysis notebook (`notebooks/02_ordinal_model`)
and the production Vertex trainer (`training/train.py`), so the loop, gradient clipping, early stopping,
best-model restore, and metrics are identical by construction — the two callers differ only in data
source (local parquet vs BigQuery), device (MPS vs CPU/CUDA), sample size, and artifact I/O.

Metric convention: MAE/RMSE on the CONTINUOUS prediction (rewards calibration of a regression head),
QWK/Accuracy/confusion on the prediction ROUNDED to the nearest star in [1, 5].
QWK (Cohen's quadratic-weighted kappa) and Spearman are the rank-agreement metrics that expose
whether the model preserves ordinal ordering — what plain RMSE hides.
Macro-F1 averages per-star F1 unweighted, so it exposes minority-class (1-2star) neglect that overall
accuracy hides on this 46%-5star corpus.

Torch/transformers are imported lazily inside the model functions so metrics-only importers stay light.
"""
import random
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error, mean_squared_error


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    import torch
    torch.manual_seed(s)


def round_star(y_cont):
    return np.clip(np.rint(np.clip(y_cont, 1, 5)), 1, 5).astype(int)


def ordinal_metrics(y_true, y_pred_cont):
    y_true = np.asarray(y_true, dtype=float)
    yc = np.clip(np.asarray(y_pred_cont, dtype=float), 1, 5)
    yr = round_star(yc)
    return {
        "MAE": mean_absolute_error(y_true, yc),
        "RMSE": mean_squared_error(y_true, yc) ** 0.5,
        "Spearman": float(spearmanr(y_true, yc).statistic),
        "QWK": cohen_kappa_score(y_true.astype(int), yr, weights="quadratic"),
        "MacroF1": float(f1_score(y_true.astype(int), yr, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)),
        "Acc": float((yr == y_true.astype(int)).mean()),
    }


def build_model(base_encoder, device):
    """Sequence-classification checkpoint with a single-output regression head, moved to `device`."""
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        base_encoder, num_labels=1, ignore_mismatched_sizes=True)
    model.config.problem_type = "regression"
    return model.to(device)


def make_encode(tokenizer, max_seq_len):
    """Return a closure that tokenizes a text iterable to padded/truncated tensors at `max_seq_len`."""
    def encode(texts):
        return tokenizer(list(texts), truncation=True, max_length=max_seq_len,
                         padding="max_length", return_tensors="pt")
    return encode


def predict(model, encode, texts, device, batch_size=64):
    """Continuous (unclipped) star predictions from the regression head, as a numpy array."""
    import torch
    texts = list(texts)
    model.eval()
    enc = encode(texts)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            ids = enc["input_ids"][i:i + batch_size].to(device)
            am = enc["attention_mask"][i:i + batch_size].to(device)
            out.append(model(input_ids=ids, attention_mask=am).logits.squeeze(-1).cpu())
    return torch.cat(out).numpy()


def train_ordinal(model, encode, train_df, val_df, device, *, epochs, batch_size, lr,
                  patience, max_grad_norm, seed=42, log=print):
    """Shared ordinal fine-tune used by BOTH the notebook and the production trainer.

    MSE regression head on the float star, with **gradient clipping**, **per-epoch validation
    monitoring** (train/val loss, val QWK, gradient norm), **early stopping**, and **best-model
    restore**. `train_df`/`val_df` need `text` and `stars` columns; the caller decides sampling,
    device, and I/O.

    Returns `(model, {"best_epoch": int, "history": [ {epoch, train_loss, val_loss, val_QWK,
    grad_mean, grad_max} ]})`. The returned model IS the object passed in — mutated in place and left
    holding the **best** epoch's weights (not the last) — but it is returned explicitly so callers can
    write `model, hist = train_ordinal(...)` and not rely on the in-place side effect.
    """
    import time
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    enc_tr = encode(train_df["text"])
    y_tr = torch.tensor(train_df["stars"].values, dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], y_tr),
                    batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = len(dl) * epochs
    warmup = min(int(0.06 * steps), len(dl))  # cap warmup near 1 epoch so a high epoch ceiling doesn't stall the ramp
    sched = get_linear_schedule_with_warmup(opt, warmup, steps)
    yv = val_df["stars"].values.astype(float)

    history = []
    best_qwk, best_epoch, best_state, stale = -1.0, 0, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        running = 0.0
        gnorms = []
        for ids, am, y in dl:
            ids, am, y = ids.to(device), am.to(device), y.to(device)
            opt.zero_grad()
            loss = model(input_ids=ids, attention_mask=am, labels=y).loss
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)))
            opt.step()
            sched.step()
            running += loss.item()
        vpred = predict(model, encode, val_df["text"], device, batch_size)
        vqwk = ordinal_metrics(yv, vpred)["QWK"]
        vloss = float(np.mean((vpred - yv) ** 2))  # unclipped MSE — comparable to the train loss
        history.append({"epoch": ep + 1, "train_loss": running / len(dl), "val_loss": vloss, "val_QWK": vqwk,
                        "grad_mean": float(np.mean(gnorms)) if gnorms else 0.0,
                        "grad_max": float(np.max(gnorms)) if gnorms else 0.0})
        tag = ""
        if vqwk > best_qwk + 1e-4:  # track the BEST epoch; its weights are restored after the loop
            best_qwk, best_epoch, stale = vqwk, ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            tag = " *best"
        else:
            stale += 1
        h = history[-1]
        log(f"epoch {ep + 1}/{epochs}: train {h['train_loss']:.3f}  val {vloss:.3f}  QWK {vqwk:.3f}  "
            f"|g| {h['grad_mean']:.2f}/{h['grad_max']:.2f}{tag}  ({time.time() - t0:.0f}s)")
        if stale >= patience:
            log(f"early stop @ epoch {ep + 1}: no val-QWK gain in {patience} epochs")
            break
    if best_state is not None:
        model.load_state_dict(best_state)  # leave the model holding the BEST epoch, not the last
    log(f"restored best model — epoch {best_epoch}, val_QWK {best_qwk:.3f}")
    return model, {"best_epoch": best_epoch, "history": history}
