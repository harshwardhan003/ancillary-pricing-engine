"""Stronger baselines than the naive control.

`tuned_rules` is what a good partner manager could achieve without ML: one flat
markup per partner and device, picked with hindsight from exact expected net
margins, within the partner's markup limits, cheapest first. It is deliberately
generous (it peeks at the truth), so beating it is a conservative test.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .simulator import Sessions, expected_outcomes, rank_from_key, sample_sessions

N_SEG = len(C.PARTNERS) * 2


def segment(s: Sessions) -> np.ndarray:
    """Segment = partner x device."""
    return s.partner * 2 + s.mobile


def allowed_by_segment(grid: np.ndarray = C.MARKUP_GRID) -> np.ndarray:
    partner = np.arange(N_SEG) // 2
    lo, hi = C.PARTNER_MARKUP_MIN[partner], C.PARTNER_MARKUP_MAX[partner]
    return (grid[None, :] >= lo[:, None] - 1e-9) & (grid[None, :] <= hi[:, None] + 1e-9)


def flat_markup_plan(s: Sessions, markup: np.ndarray):
    price = s.cost * (1 + markup[:, None])
    return price, rank_from_key(price)


def segment_markup_table(s: Sessions, grid: np.ndarray = C.MARKUP_GRID):
    """Exact expected net margin per session for each (segment, flat markup)."""
    seg = segment(s)
    table = np.zeros((N_SEG, len(grid)))
    for j, m in enumerate(grid):
        price, rank = flat_markup_plan(s, np.full(len(s), m))
        _, ev = expected_outcomes(s, price, rank)
        for k in range(N_SEG):
            table[k, j] = ev[seg == k].mean() if (seg == k).any() else 0.0
    weights = np.array([(seg == k).mean() for k in range(N_SEG)])
    return table, weights


def fit_tuned_rules(seed: int = 1, n: int = 80_000, grid: np.ndarray = C.MARKUP_GRID):
    s = sample_sessions(n, np.random.default_rng(seed))
    table, _ = segment_markup_table(s, grid)
    table = np.where(allowed_by_segment(grid), table, -np.inf)
    return grid[table.argmax(axis=1)]


def tuned_rules_policy(best_markup_by_segment: np.ndarray):
    def policy(s: Sessions):
        return flat_markup_plan(s, best_markup_by_segment[segment(s)])

    return policy
