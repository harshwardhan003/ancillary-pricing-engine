"""Offline A/B test harness.

Sessions are randomly split 50/50, each arm is served by its own policy, and
outcomes are drawn from the true (hidden) customer model. We report observed
statistics with confidence intervals, plus the exact expected uplift computed
analytically on the same sessions as a variance-free cross-check.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .simulator import Sessions, expected_outcomes, realised, sample_choice, true_choice_probs


def _arm_stats(booked, margin, gbv):
    return {
        "sessions": len(booked),
        "conversion": float(booked.mean()),
        "margin_per_session": float(margin.mean()),
        "gbv_per_session": float(gbv.mean()),
        "margin_per_booking": float(margin[booked].mean()) if booked.any() else 0.0,
    }


def run_ab(s: Sessions, policy_a, policy_b, rng: np.random.Generator) -> dict:
    n = len(s)
    in_b = rng.random(n) < 0.5
    price_a, rank_a = policy_a(s)
    price_b, rank_b = policy_b(s)
    price = np.where(in_b[:, None], price_b, price_a)
    rank = np.where(in_b[:, None], rank_b, rank_a)

    chosen = sample_choice(true_choice_probs(s, price, rank), rng)
    booked, margin, gbv = realised(s, price, chosen)
    a, b = ~in_b, in_b

    out = {"control": _arm_stats(booked[a], margin[a], gbv[a]),
           "treatment": _arm_stats(booked[b], margin[b], gbv[b])}

    # Margin per session: relative uplift with delta-method CI and Welch test.
    ma, mb = margin[a].mean(), margin[b].mean()
    sea, seb = margin[a].std(ddof=1) / np.sqrt(a.sum()), margin[b].std(ddof=1) / np.sqrt(b.sum())
    uplift = mb / ma - 1
    se_rel = np.sqrt((seb / ma) ** 2 + (mb * sea / ma**2) ** 2)
    t = stats.ttest_ind(margin[b], margin[a], equal_var=False)
    out["margin_uplift"] = {
        "relative": float(uplift),
        "ci95": [float(uplift - 1.96 * se_rel), float(uplift + 1.96 * se_rel)],
        "p_value": float(t.pvalue),
    }

    # Conversion: two-proportion z-test.
    pa, pb = booked[a].mean(), booked[b].mean()
    pool = booked.mean()
    se = np.sqrt(pool * (1 - pool) * (1 / a.sum() + 1 / b.sum()))
    z = (pb - pa) / se
    se_rel_c = np.sqrt(pb * (1 - pb) / (b.sum() * pa**2) + (pb**2) * pa * (1 - pa) / (a.sum() * pa**4))
    rel_c = pb / pa - 1
    out["conversion_change"] = {
        "relative": float(rel_c),
        "ci95": [float(rel_c - 1.96 * se_rel_c), float(rel_c + 1.96 * se_rel_c)],
        "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
    }

    # Exact expectation on all sessions (no sampling noise).
    ca, xa = expected_outcomes(s, price_a, rank_a)
    cb, xb = expected_outcomes(s, price_b, rank_b)
    out["analytic"] = {
        "margin_uplift": float(xb.mean() / xa.mean() - 1),
        "conversion_change": float(cb.mean() / ca.mean() - 1),
        "control_conversion": float(ca.mean()),
        "treatment_conversion": float(cb.mean()),
        "control_margin_per_session": float(xa.mean()),
        "treatment_margin_per_session": float(xb.mean()),
    }
    return out
