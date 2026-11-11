"""Per-partner calibration of the booking value `lam`.

Each partner has its own tolerance for lost bookings (conv_floor, an assumption).
For every partner we sweep `lam`, and pick the one with the highest expected net
margin uplift whose booking-rate change vs the control stays above -conv_floor.

    python -m ancillary.calibrate --artifacts artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as C
from .model import ConversionModel
from .optimiser import optimise
from .simulator import TrafficMix, control_policy, expected_outcomes, sample_sessions

LAMS = [0, 5, 10, 15, 20, 30, 40, 60]


def calibrate_partner(model, i: int, n: int, seed: int) -> dict:
    s = sample_sessions(n, np.random.default_rng(seed + i), TrafficMix.only(i))
    cp, cr = control_policy(s)
    c0, m0 = (a.mean() for a in expected_outcomes(s, cp, cr))
    sweep = []
    for lam in LAMS:
        plan = optimise(model, s, lam=float(lam))
        c, m = (a.mean() for a in expected_outcomes(s, plan.price, plan.rank))
        sweep.append({"lam": lam, "margin_uplift": float(m / m0 - 1),
                      "conversion_change": float(c / c0 - 1), "mean_markup": float(plan.markup.mean())})
    floor = float(C.PARTNER_CONV_FLOOR[i])
    ok = [r for r in sweep if r["conversion_change"] >= -floor]
    best = max(ok, key=lambda r: r["margin_uplift"]) if ok else max(sweep, key=lambda r: r["conversion_change"])
    return {"lam": float(best["lam"]), "markup_min": float(C.PARTNER_MARKUP_MIN[i]),
            "markup_max": float(C.PARTNER_MARKUP_MAX[i]), "conv_floor": floor,
            "share": float(C.PARTNER_SHARE[i]), "kind": C.PARTNER_KIND[i],
            "slug": C.PARTNER_SLUG[i], "chosen": best, "sweep": sweep}


def calibrate(model, n: int = 20_000, seed: int = 700) -> dict:
    return {"model_version": model.version,
            "partners": {C.PARTNERS[i]: calibrate_partner(model, i, n, seed) for i in range(len(C.PARTNERS))}}


def lam_vector(cfg: dict) -> np.ndarray:
    return np.array([cfg["partners"][p]["lam"] for p in C.PARTNERS])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    cfg = calibrate(ConversionModel.load(art / "model.json"), 3_000 if a.quick else 20_000)
    (art / "partner_config.json").write_text(json.dumps(cfg, indent=2))
    for p, v in cfg["partners"].items():
        c = v["chosen"]
        print(f"{p:20s} lam={v['lam']:5.1f} margin {c['margin_uplift']:+.1%} bookings {c['conversion_change']:+.1%} (floor -{v['conv_floor']:.0%})")


if __name__ == "__main__":
    main()
