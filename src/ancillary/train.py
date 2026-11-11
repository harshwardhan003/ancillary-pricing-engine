"""Train the conversion model on simulated exploration logs.

    python -m ancillary.train --sessions 150000 --out artifacts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config as C
from .features import FEATURE_NAMES, features_from_sessions
from .mlp import MLPModel
from .model import ConversionModel
from .monitoring import DriftMonitor
from .simulator import logging_policy, sample_choice, sample_sessions, true_choice_probs


def make_dataset(n_sessions: int, seed: int):
    rng = np.random.default_rng(seed)
    s = sample_sessions(n_sessions, rng)
    price, rank = logging_policy(s, rng)
    chosen = sample_choice(true_choice_probs(s, price, rank), rng)
    X = features_from_sessions(s, price, rank)  # (n,K,F)
    y = chosen[:, None] == np.arange(C.N_OFFERS)[None, :]
    return s, X, y.astype(int)


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    y, p = y[order], p[order]
    total = 0.0
    for idx in np.array_split(np.arange(len(p)), bins):
        total += len(idx) * abs(y[idx].mean() - p[idx].mean())
    return float(total / len(p))


def metrics(y, p) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "mean_pred": float(p.mean()),
        "mean_obs": float(y.mean()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--skip-gbm", action="store_true", help="skip the benchmark model")
    ap.add_argument("--skip-mlp", action="store_true", help="skip the neural network challenger")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    s, X, y = make_dataset(args.sessions, args.seed)
    cut = int(0.8 * len(s))
    F = X.shape[-1]
    Xtr, ytr = X[:cut].reshape(-1, F), y[:cut].reshape(-1)
    Xte, yte = X[cut:].reshape(-1, F), y[cut:].reshape(-1)

    model = ConversionModel.fit(Xtr, ytr, meta={
        "version": C.MODEL_VERSION, "trained_sessions": int(cut),
        "data": "synthetic exploration logs", "seed": args.seed})
    report = {"rows": {"train": len(ytr), "test": len(yte),
                       "positive_rate": float(y.mean())},
              "logistic_regression": metrics(yte, model.predict(Xte))}

    if not args.skip_gbm:
        from sklearn.ensemble import HistGradientBoostingClassifier

        gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        gbm.fit(Xtr, ytr)
        report["gradient_boosting"] = metrics(yte, gbm.predict_proba(Xte)[:, 1])
    if not args.skip_mlp:
        mlp = MLPModel.fit(Xtr, ytr, meta={"version": C.MODEL_VERSION + "-mlp", "seed": args.seed})
        report["mlp"] = metrics(yte, mlp.predict(Xte))
        mlp.save(out / "model_mlp.json")
    report["constant_rate_baseline"] = metrics(yte, np.full(len(yte), ytr.mean()))

    w = dict(zip(FEATURE_NAMES, model.w))
    report["key_coefficients"] = {k: float(w[k]) for k in
                                  ("rel_price_ref", "rel_price_set", "ln_rank", "rating_c", "cap_gap")}
    scored = {k: v["log_loss"] for k, v in report.items() if isinstance(v, dict) and "log_loss" in v}
    report["best_by_log_loss"] = min(scored, key=scored.get)
    report["train_seconds"] = round(time.time() - t0, 1)

    model.meta["test_metrics"] = report["logistic_regression"]
    model.save(out / "model.json")
    DriftMonitor.fit(s.take(slice(0, cut))).save(out / "drift_reference.json")
    (out / "train_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
