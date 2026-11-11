"""A small neural network challenger to the logistic regression.

Trained with scikit-learn, then exported to plain JSON so serving needs only
numpy. It exposes the same `predict(X)` as ConversionModel, so the optimiser
can use either one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


class MLPModel:
    def __init__(self, mu, sd, layers, meta: dict | None = None):
        self.mu, self.sd = np.asarray(mu, float), np.asarray(sd, float)
        self.layers = [(np.asarray(w, float), np.asarray(b, float)) for w, b in layers]
        self.meta = meta or {}

    @property
    def version(self) -> str:
        return self.meta.get("version", "unknown")

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = (X - self.mu) / self.sd
        for w, b in self.layers[:-1]:
            h = np.maximum(h @ w + b, 0.0)  # ReLU
        w, b = self.layers[-1]
        z = (h @ w + b)[..., 0]
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def fit(cls, X, y, hidden=(32, 16), seed: int = 0, max_epochs: int = 40,
            patience: int = 4, meta: dict | None = None):
        """Early stopping on validation LOG LOSS (sklearn's built-in one uses accuracy,
        which is useless when only ~1% of rows are bookings)."""
        import copy

        from sklearn.metrics import log_loss
        from sklearn.neural_network import MLPClassifier

        rng = np.random.default_rng(seed)
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - mu) / sd
        idx = rng.permutation(len(Z))
        n_val = len(Z) // 10
        va, tr = idx[:n_val], idx[n_val:]
        net = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-3, learning_rate_init=1e-3,
                            random_state=seed)
        best, best_net, bad = np.inf, None, 0
        for _ in range(max_epochs):
            order = rng.permutation(tr)
            for chunk in np.array_split(order, max(1, len(order) // 2048)):
                net.partial_fit(Z[chunk], y[chunk], classes=[0, 1])
            loss = log_loss(y[va], net.predict_proba(Z[va])[:, 1])
            if loss < best - 1e-6:
                best, best_net, bad = loss, copy.deepcopy(net), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        layers = list(zip(best_net.coefs_, best_net.intercepts_))
        return cls(mu, sd, layers, meta)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "kind": "mlp", "feature_names": FEATURE_NAMES, "mu": self.mu.tolist(),
            "sd": self.sd.tolist(), "layers": [[w.tolist(), b.tolist()] for w, b in self.layers],
            "meta": self.meta}))

    @classmethod
    def load(cls, path) -> MLPModel:
        d = json.loads(Path(path).read_text())
        if d["feature_names"] != FEATURE_NAMES:
            raise ValueError("model was trained with a different feature set")
        return cls(d["mu"], d["sd"], d["layers"], d.get("meta", {}))
