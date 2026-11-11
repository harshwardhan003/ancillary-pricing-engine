"""Run every experiment reported in the README and write artifacts/results.json.

    python -m ancillary.evaluate --out artifacts
    python -m ancillary.evaluate --quick      # small sizes, used by CI
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config as C
from .abtest import run_ab
from .bandit import simulate_bandit
from .baselines import fit_tuned_rules, tuned_rules_policy
from .calibrate import lam_vector
from .features import features_from_sessions
from .model import ConversionModel
from .monitoring import DriftMonitor
from .optimiser import optimise
from .simulator import (
    TrafficMix,
    control_policy,
    expected_gbv,
    expected_outcomes,
    partner_commission,
    sample_sessions,
)
from .train import make_dataset

LAM_DEFAULT = 10.0  # single-number lam used only in the ablation ladder


def _expected(s, price, rank):
    conv, margin = expected_outcomes(s, price, rank)
    return float(conv.mean()), float(margin.mean())


def ladder(model, s, tuned):
    """Where does the uplift come from? Analytic expectations on the same sessions."""
    rows = {}
    cp, cr = control_policy(s)
    rows["control"] = _expected(s, cp, cr)
    tp, tr = tuned(s)
    rows["tuned_rules"] = _expected(s, tp, tr)
    for name, kw in {
        "ml_markup_only": {"tune_markup": True, "tune_sort": False},
        "ml_sort_only": {"tune_markup": False, "tune_sort": True},
        "ml_markup_and_sort": {"tune_markup": True, "tune_sort": True},
    }.items():
        plan = optimise(model, s, lam=LAM_DEFAULT, **kw)
        rows[name] = _expected(s, plan.price, plan.rank)
    c0, m0 = rows["control"]
    mt = rows["tuned_rules"][1]
    return {
        k: {"conversion": c, "margin_per_session": m,
            "margin_uplift_vs_control": m / m0 - 1,
            "margin_uplift_vs_tuned_rules": m / mt - 1,
            "conversion_change_vs_control": c / c0 - 1}
        for k, (c, m) in rows.items()
    }


def scorecard(model, s, lam_vec, tuned):
    """Per-partner results on the same sessions (exact expectations, no sampling noise)."""
    cp, cr = control_policy(s)
    cc, cm = expected_outcomes(s, cp, cr)
    cg = expected_gbv(s, cp, cr)
    plan = optimise(model, s, lam=lam_vec)
    mc, mm = expected_outcomes(s, plan.price, plan.rank)
    mg = expected_gbv(s, plan.price, plan.rank)
    tp, tr = tuned(s)
    _, tm = expected_outcomes(s, tp, tr)
    rows = {}
    for i, name in enumerate(C.PARTNERS):
        k = s.partner == i
        r = {
            "sessions": int(k.sum()), "lam": float(lam_vec[i]),
            "control": {"booking_rate": float(cc[k].mean()), "net_margin": float(cm[k].mean()),
                        "partner_commission": float(partner_commission(s, cm)[k].mean()), "gbv": float(cg[k].mean())},
            "ml": {"booking_rate": float(mc[k].mean()), "net_margin": float(mm[k].mean()),
                   "partner_commission": float(partner_commission(s, mm)[k].mean()), "gbv": float(mg[k].mean()),
                   "mean_markup": float(plan.markup[k].mean())},
        }
        r["margin_uplift"] = r["ml"]["net_margin"] / r["control"]["net_margin"] - 1
        r["booking_change"] = r["ml"]["booking_rate"] / r["control"]["booking_rate"] - 1
        r["commission_change"] = r["ml"]["partner_commission"] / r["control"]["partner_commission"] - 1
        r["margin_uplift_vs_tuned_rules"] = r["ml"]["net_margin"] / float(tm[k].mean()) - 1
        r["conv_floor"] = float(C.PARTNER_CONV_FLOOR[i])
        r["guardrail_pass"] = bool(r["booking_change"] >= -r["conv_floor"])
        rows[name] = r
    return rows


def calibration_table(model, n_sessions, seed):
    _, X, y = make_dataset(n_sessions, seed)
    p = model.predict(X.reshape(-1, X.shape[-1]))
    y = y.reshape(-1)
    order = np.argsort(p)
    rows = []
    for idx in np.array_split(order, 10):
        rows.append({"mean_pred": float(p[idx].mean()), "mean_obs": float(y[idx].mean())})
    return rows


def drift_demo(model, monitor, lam, n):
    """lam: per-partner vector."""
    # Airbnb and eDreams traffic surges, bookings come much later, more mobile.
    shifted = TrafficMix(partner_probs=(0.08, 0.07, 0.05, 0.03, 0.40, 0.37), lead_mult=0.3, mobile_add=0.2)
    res = {}
    for name, mix in {"in_distribution": TrafficMix(), "shifted": shifted}.items():
        s = sample_sessions(n, np.random.default_rng(500 + len(name)), mix)
        report = monitor.check(s)
        cp, cr = control_policy(s)
        c0, m0 = _expected(s, cp, cr)
        plan = optimise(model, s, lam=lam)
        c1, m1 = _expected(s, plan.price, plan.rank)
        res[name] = {"drift": report, "margin_uplift_vs_control": m1 / m0 - 1,
                     "conversion_change_vs_control": c1 / c0 - 1}
    return res


def demo_fixtures(model, lam, seed=2025, n=10):
    """A few sessions with Python outputs, so the browser demo can be cross-checked."""
    s = sample_sessions(n, np.random.default_rng(seed))
    _cp, cr = control_policy(s)
    plan = optimise(model, s, lam=lam)
    X = features_from_sessions(s, plan.price, plan.rank)
    rows = []
    for i in range(n):
        rows.append({
            "session": {"partner": int(s.partner[i]), "market": int(s.market[i]), "mobile": int(s.mobile[i]),
                        "returning": int(s.returning[i]), "lead": float(s.lead[i]),
                        "trip": int(s.trip[i]), "party": int(s.party[i])},
            "offers": [{"sup": int(s.sup[i, k]), "cls": int(s.cls[i, k]),
                        "rating": float(s.rating[i, k]), "cost": float(s.cost[i, k]),
                        "ref": float(s.ref[i, k])} for k in range(C.N_OFFERS)],
            "control_rank": cr[i].tolist(),
            "ml_markup": plan.markup[i].tolist(), "ml_rank": plan.rank[i].tolist(),
            "ml_p": plan.p_book[i].tolist(), "ml_features_row0": X[i, 0].tolist(),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    art = Path(args.artifacts)
    model = ConversionModel.load(art / "model.json")
    monitor = DriftMonitor.load(art / "drift_reference.json")

    q = args.quick
    n_ab, n_lad, n_score, n_drift = (6_000, 4_000, 6_000, 4_000) if q else (60_000, 20_000, 40_000, 20_000)
    t0 = time.time()
    pcfg = json.loads((art / "partner_config.json").read_text())
    lam_vec = lam_vector(pcfg)
    res = {"model_version": model.version, "lam_default": LAM_DEFAULT,
           "lam_by_partner": dict(zip(C.PARTNERS, lam_vec.tolist())), "quick": q}

    tuned = tuned_rules_policy(fit_tuned_rules(n=8_000 if q else 60_000))

    # 1) Portfolio A/B tests with per-partner lam: ML engine vs naive control and vs tuned rules.
    s = sample_sessions(n_ab, np.random.default_rng(2024))
    plan = optimise(model, s, lam=lam_vec)
    ml = lambda _s: (plan.price, plan.rank)
    res["ab_vs_control"] = run_ab(s, control_policy, ml, np.random.default_rng(1))
    res["ab_vs_tuned_rules"] = run_ab(s, tuned, ml, np.random.default_rng(2))
    res["aa_test"] = run_ab(s, control_policy, control_policy, np.random.default_rng(3))
    print(f"A/B done {time.time() - t0:.0f}s", flush=True)

    # 2) Partner scorecard.
    s2 = sample_sessions(n_score, np.random.default_rng(13))
    res["scorecard"] = scorecard(model, s2, lam_vec, tuned)
    print(f"scorecard done {time.time() - t0:.0f}s", flush=True)

    # 3) Where does the uplift come from?
    s3 = sample_sessions(n_lad, np.random.default_rng(11))
    res["ladder"] = ladder(model, s3, tuned)
    print(f"ladder done {time.time() - t0:.0f}s", flush=True)

    # 4) Calibration, drift, bandit onboarding, demo fixtures.
    res["calibration"] = calibration_table(model, 3_000 if q else 40_000, 99)
    res["drift_demo"] = drift_demo(model, monitor, lam_vec, n_drift)
    res["bandit"] = simulate_bandit(total=20_000 if q else 120_000, batch=1_000,
                                    mix=TrafficMix.only(C.PARTNERS.index("Airbnb")))
    res["bandit_partner"] = "Airbnb"
    res["demo_fixtures"] = demo_fixtures(model, lam_vec)
    res["runtime_seconds"] = round(time.time() - t0, 1)

    (art / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("demo_fixtures", "calibration")}, indent=2)[:7000])


if __name__ == "__main__":
    main()
