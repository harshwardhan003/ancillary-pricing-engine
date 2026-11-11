"""Margin-aware price and sort optimiser.

For every session we choose a markup per offer (from a bounded grid) and an
ordering that maximise

    sum_k  p_k * (net_margin_k + lam)

where p_k is the model's booking probability for offer k in its position,
net_margin_k is CarTrawler's margin after the partner's revenue share, and `lam`
is a value placed on a booking itself. Each partner gets its own `lam` (a
guardrail that trades margin for bookings) and its own markup floor and cap.

The search is coordinate ascent over offers, vectorised across sessions, with
an expected-value re-sort between passes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .features import build_features
from .model import ConversionModel
from .simulator import Sessions, rank_from_key


@dataclass
class Plan:
    price: np.ndarray  # (n,K) price per day
    markup: np.ndarray  # (n,K)
    rank: np.ndarray  # (n,K) 0 = top
    p_book: np.ndarray  # (n,K) model booking probability
    exp_margin: np.ndarray  # (n,K) p * CarTrawler net margin if booked


def _score(model, ctx, s, price, rank):
    X = build_features(*ctx, s.cls, s.rating, price, s.ref, rank)
    return model.predict(X)


def _lam_per_session(lam, s: Sessions) -> np.ndarray:
    """lam may be one number, or one number per partner."""
    a = np.asarray(lam, dtype=float)
    return np.full(len(s), float(a)) if a.ndim == 0 else a[s.partner]


def _optimise_chunk(model: ConversionModel, s: Sessions, lam, grid: np.ndarray, passes: int,
                    tune_markup: bool, tune_sort: bool) -> Plan:
    n, k = s.cost.shape
    ctx = s.ctx()
    trip = s.trip[:, None].astype(float)
    net = (1.0 - C.PARTNER_SHARE[s.partner])[:, None]
    lam_v = _lam_per_session(lam, s)[:, None]
    lo, hi = C.PARTNER_MARKUP_MIN[s.partner], C.PARTNER_MARKUP_MAX[s.partner]
    allowed = (grid[:, None] >= lo[None, :] - 1e-9) & (grid[:, None] <= hi[None, :] + 1e-9)  # (g,n)
    markup = np.full((n, k), C.BASELINE_MARKUP)
    rank = rank_from_key(s.cost * (1 + markup))
    g = len(grid)

    for _ in range(passes):
        if tune_markup:
            for i in range(k):
                cand = np.broadcast_to(markup, (g, n, k)).copy()
                cand[:, :, i] = grid[:, None]
                price = s.cost[None] * (1 + cand)
                p = _score(model, ctx, s, price, rank)
                value = (p * (net[None] * s.cost[None] * cand * trip[None] + lam_v[None])).sum(axis=-1)
                value = np.where(allowed, value, -np.inf)  # respect the partner's markup limits
                markup[:, i] = grid[value.argmax(axis=0)]
        price = s.cost * (1 + markup)
        if tune_sort:
            p = _score(model, ctx, s, price, rank)
            rank = rank_from_key(-(p * (net * s.cost * markup * trip + lam_v)))
        else:
            rank = rank_from_key(price)  # cheapest first, as in the control

    price = s.cost * (1 + markup)
    p = _score(model, ctx, s, price, rank)
    return Plan(price, markup, rank, p, p * net * s.cost * markup * trip)


def optimise(model: ConversionModel, s: Sessions, lam=0.0,
             grid: np.ndarray = C.MARKUP_GRID, passes: int = 3, chunk: int = 4000,
             tune_markup: bool = True, tune_sort: bool = True) -> Plan:
    """lam: one number, or one per partner. tune_markup / tune_sort switch the two
    levers on and off (used for ablations)."""
    parts = [
        _optimise_chunk(model, s.take(slice(a, a + chunk)), lam, grid, passes,
                        tune_markup, tune_sort)
        for a in range(0, len(s), chunk)
    ]
    return Plan(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                  ("price", "markup", "rank", "p_book", "exp_margin")))


def ml_policy(model: ConversionModel, lam=0.0, **kw):
    def policy(s: Sessions):
        plan = optimise(model, s, lam=lam, **kw)
        return plan.price, plan.rank

    return policy
