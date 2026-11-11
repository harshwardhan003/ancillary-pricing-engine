"""Conversion model.

A regularised logistic regression whose weights are stored as plain JSON. That
keeps the serving image tiny (numpy only), makes the model auditable, and lets
the browser demo run the exact same model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


class ConversionModel:
    def __init__(self, w: np.ndarray, b: float, meta: dict | None = None):
        self.w = np.asarray(w, dtype=float)
        self.b = float(b)
        self.meta = meta or {}
        if len(self.w) != len(FEATURE_NAMES):
            raise ValueError("weight vector does not match FEATURE_NAMES")

    @property
    def version(self) -> str:
        return self.meta.get("version", "unknown")

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, C: float = 1.0, meta: dict | None = None):
        from sklearn.linear_model import LogisticRegression

        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=1000)
        lr.fit((X - mu) / sd, y)
        coef = lr.coef_[0]
        w_raw = coef / sd
        b_raw = float(lr.intercept_[0] - np.sum(coef * mu / sd))
        return cls(w_raw, b_raw, meta)

    def to_dict(self) -> dict:
        return {
            "feature_names": FEATURE_NAMES,
            "weights": self.w.tolist(),
            "intercept": self.b,
            "meta": self.meta,
        }

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> ConversionModel:
        d = json.loads(Path(path).read_text())
        if d["feature_names"] != FEATURE_NAMES:
            raise ValueError("model was trained with a different feature set")
        return cls(np.array(d["weights"]), d["intercept"], d.get("meta", {}))
