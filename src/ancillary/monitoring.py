"""Data drift monitoring with the Population Stability Index (PSI).

Rule of thumb used by most revenue and credit teams:
PSI < 0.10 stable, 0.10 to 0.25 moderate shift, > 0.25 major shift (investigate or retrain).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .simulator import Sessions

WARN, ALERT = 0.10, 0.25
_NUMERIC = {"lead_days": lambda s: s.lead, "trip_days": lambda s: s.trip,
            "mean_cost_per_day": lambda s: s.cost.mean(axis=1)}
_CATEG = {"partner_mix": lambda s: s.partner, "market": lambda s: s.market,
          "mobile": lambda s: s.mobile, "returning": lambda s: s.returning,
          "party_size": lambda s: s.party}


def psi(ref: np.ndarray, cur: np.ndarray, eps: float = 1e-4) -> float:
    ref = np.clip(np.asarray(ref, float), eps, None)
    cur = np.clip(np.asarray(cur, float), eps, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _status(v: float) -> str:
    return "alert" if v > ALERT else "warn" if v > WARN else "ok"


class DriftMonitor:
    def __init__(self, reference: dict):
        self.ref = reference

    @classmethod
    def fit(cls, s: Sessions, n_bins: int = 10) -> DriftMonitor:
        ref = {"numeric": {}, "categorical": {}}
        for name, fn in _NUMERIC.items():
            x = fn(s)
            edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
            counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
            ref["numeric"][name] = {"edges": edges.tolist(), "p": (counts / counts.sum()).tolist()}
        for name, fn in _CATEG.items():
            x = fn(s)
            counts = np.bincount(x, minlength=int(x.max()) + 1 + 1)
            ref["categorical"][name] = {"p": (counts / counts.sum()).tolist()}
        return cls(ref)

    def check(self, s: Sessions) -> dict:
        report = {}
        for name, fn in _NUMERIC.items():
            r = self.ref["numeric"][name]
            idx = np.searchsorted(np.array(r["edges"]), fn(s), side="right")
            cur = np.bincount(idx, minlength=len(r["p"]))
            v = psi(np.array(r["p"]), cur)
            report[name] = {"psi": round(v, 4), "status": _status(v)}
        for name, fn in _CATEG.items():
            ref_p = np.array(self.ref["categorical"][name]["p"])
            cur = np.bincount(fn(s), minlength=len(ref_p))[: len(ref_p)]
            v = psi(ref_p, cur)
            report[name] = {"psi": round(v, 4), "status": _status(v)}
        report["_overall"] = max((v["status"] for v in report.values()),
                                 key=["ok", "warn", "alert"].index)
        return report

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.ref, indent=2))

    @classmethod
    def load(cls, path) -> DriftMonitor:
        return cls(json.loads(Path(path).read_text()))
