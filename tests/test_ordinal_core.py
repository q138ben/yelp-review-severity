"""Parity test for the shared ordinal training core (src/ordinal.py).

notebooks/02_ordinal_model.py and training/train.py BOTH import build_model / make_encode /
train_ordinal / predict from this one module, so exercising it here proves the training procedure they
run is identical by construction. Run either way:
    .venv/bin/python -m pytest tests/test_ordinal_core.py
    .venv/bin/python tests/test_ordinal_core.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ordinal import ordinal_metrics, build_model, make_encode, predict, train_ordinal


def test_ordinal_metrics_perfect_and_keys():
    y = [1, 2, 3, 4, 5, 5, 1]
    m = ordinal_metrics(y, y)
    assert set(m) == {"MAE", "RMSE", "Spearman", "QWK", "MacroF1", "Acc"}
    assert m["MAE"] == 0.0 and round(m["QWK"], 6) == 1.0 and m["Acc"] == 1.0


def test_ordinal_metrics_penalises_far_miss_more():
    y = [1, 1, 5, 5]
    near = ordinal_metrics(y, [2, 2, 4, 4])   # every prediction off by 1
    far = ordinal_metrics(y, [5, 5, 1, 1])    # rank-reversed, off by 4
    assert far["QWK"] < near["QWK"]
    assert far["MAE"] > near["MAE"]


def test_train_ordinal_contract():
    """Smoke-run the shared loop on a tiny fixture: history schema, best-model restore, predict shape."""
    import torch
    from transformers import AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    base = "distilbert-base-uncased-finetuned-sst-2-english"
    pos = ["great food loved it", "amazing service best ever", "wonderful perfect delightful"]
    neg = ["terrible slow and rude", "awful cold disgusting", "worst experience never again"]
    df = pd.DataFrame({"text": pos * 8 + neg * 8, "stars": [5, 5, 5] * 8 + [1, 1, 1] * 8})
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    train_df, val_df = df.iloc[:36], df.iloc[36:]

    tok = AutoTokenizer.from_pretrained(base)
    encode = make_encode(tok, 32)
    built = build_model(base, dev)
    p_before = built.state_dict()["pre_classifier.weight"].detach().cpu().clone()
    model, hist = train_ordinal(built, encode, train_df, val_df, dev,
                                epochs=3, batch_size=8, lr=2e-5, patience=5, max_grad_norm=1.0,
                                log=lambda m: None)

    assert model is built                                        # trained in place AND returned explicitly
    p_after = model.state_dict()["pre_classifier.weight"].detach().cpu().clone()
    assert (p_before != p_after).any()                          # weights actually updated
    assert set(hist) == {"best_epoch", "history"}
    assert 1 <= hist["best_epoch"] <= 3
    h0 = hist["history"][0]
    assert set(h0) == {"epoch", "train_loss", "val_loss", "val_QWK", "grad_mean", "grad_max"}
    assert h0["grad_max"] >= h0["grad_mean"] > 0                # clipping saw real pre-clip gradients
    preds = predict(model, encode, val_df["text"], dev)
    assert preds.shape == (len(val_df),) and np.isfinite(preds).all()
    # the returned model reproduces the BEST epoch's recorded val QWK (best-model restore worked)
    live_qwk = ordinal_metrics(val_df["stars"], preds)["QWK"]
    assert abs(live_qwk - hist["history"][hist["best_epoch"] - 1]["val_QWK"]) < 1e-9


if __name__ == "__main__":
    test_ordinal_metrics_perfect_and_keys()
    test_ordinal_metrics_penalises_far_miss_more()
    test_train_ordinal_contract()
    print("all parity tests passed")
