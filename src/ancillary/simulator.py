"""Synthetic booking-session simulator.

A session is a traveller on a PARTNER's channel (an airline app, an OTA, a
marketplace) who has just booked and is shown K car rental offers. The customer picks at most one offer, following a multinomial logit
(MNL) model with an outside option ("no car"). The true utility depends on price
relative to a market reference price, on position in the list, on supplier
rating, on whether the car fits the party, and on a latent price sensitivity
that varies by market, device and customer type.

The ML model never sees the latent sensitivity or the true coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class TrafficMix:
    """Which partners send traffic, and how it shifts. Change it to simulate drift."""

    partner_probs: tuple = tuple(C.PARTNER_TRAFFIC)
    lead_mult: float = 1.0  # multiplies every partner's booking lead time
    mobile_add: float = 0.0  # added to every partner's mobile share
    party_probs: tuple = (0.28, 0.40, 0.14, 0.13, 0.05)

    @classmethod
    def only(cls, partner: int) -> TrafficMix:
        """Traffic from a single partner."""
        probs = [0.0] * len(C.PARTNERS)
        probs[partner] = 1.0
        return cls(partner_probs=tuple(probs))


@dataclass
class Sessions:
    lead: np.ndarray  # (n,) days
    trip: np.ndarray  # (n,) rental days
    party: np.ndarray  # (n,)
    market: np.ndarray  # (n,) int
    mobile: np.ndarray  # (n,) 0/1
    returning: np.ndarray  # (n,) 0/1
    partner: np.ndarray  # (n,) int, index into config.PARTNERS
    month: np.ndarray  # (n,) 1..12
    cls: np.ndarray  # (n,K) int
    sup: np.ndarray  # (n,K) int
    rating: np.ndarray  # (n,K)
    cost: np.ndarray  # (n,K) supplier cost per day
    ref: np.ndarray  # (n,K) market reference price per day
    _extra: dict = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.lead)

    def take(self, idx) -> Sessions:
        return Sessions(
            *(getattr(self, f)[idx] for f in _FIELDS),
        )

    def ctx(self):
        """Session-level arrays shaped (n,1) for broadcasting against (n,K)."""
        return tuple(
            getattr(self, f)[:, None]
            for f in ("lead", "trip", "party", "market", "mobile", "returning", "partner")
        )


_FIELDS = (
    "lead", "trip", "party", "market", "mobile", "returning", "partner", "month",
    "cls", "sup", "rating", "cost", "ref",
)


def sample_sessions(n: int, rng: np.random.Generator, mix: TrafficMix | None = None,
                    k: int = C.N_OFFERS) -> Sessions:
    mix = mix or TrafficMix()
    partner = rng.choice(len(C.PARTNERS), size=n, p=np.asarray(mix.partner_probs))
    # Market: drawn from the partner's own market profile.
    cum = np.cumsum(C.PARTNER_MARKET_PROBS[partner], axis=1)
    market = np.minimum((rng.random(n)[:, None] > cum).sum(axis=1), len(C.MARKETS) - 1)
    mobile_p = np.clip(C.PARTNER_MOBILE[partner] + mix.mobile_add, 0.0, 0.95)
    mobile = (rng.random(n) < mobile_p).astype(int)
    returning = (rng.random(n) < C.PARTNER_RETURNING[partner]).astype(int)
    lead = np.clip(np.round(rng.gamma(2.0, C.PARTNER_LEAD_SCALE[partner] * mix.lead_mult)), 0, 330)
    trip = np.clip(1 + rng.poisson(4.5, n), 1, 21)
    party = rng.choice([1, 2, 3, 4, 5], size=n, p=mix.party_probs)
    month = rng.integers(1, 13, n)

    n_combo = len(C.SUPPLIERS) * len(C.CAR_CLASSES)
    combo = np.argsort(rng.random((n, n_combo)), axis=1)[:, :k]
    sup = combo // len(C.CAR_CLASSES)
    cls = combo % len(C.CAR_CLASSES)

    season = 1.0 + 0.25 * np.sin(2 * np.pi * (month - 4) / 12)
    cost = (
        C.CLASS_BASE_COST[cls]
        * C.SUPPLIER_COST_MULT[sup]
        * C.MARKET_PRICE_MULT[market][:, None]
        * season[:, None]
        * np.exp(rng.normal(0, 0.05, (n, k)))
    )
    ref = cost * (1 + C.BASELINE_MARKUP)
    return Sessions(lead, trip, party, market, mobile, returning, partner, month, cls, sup,
                    C.SUPPLIER_RATING[sup], cost, ref)


# ----------------------------------------------------------------------------
# Ground truth customer behaviour (hidden from the model)
# ----------------------------------------------------------------------------
def rank_from_key(key: np.ndarray) -> np.ndarray:
    """Position (0 = top) of each offer when sorting ascending by `key`."""
    return np.argsort(np.argsort(key, axis=-1, kind="stable"), axis=-1, kind="stable")


def true_sensitivity(s: Sessions) -> np.ndarray:
    return (
        C.MARKET_SENS[s.market]
        * C.PARTNER_SENS[s.partner]
        * np.where(s.mobile == 1, 1.15, 1.0)
        * np.where(s.returning == 1, 0.85, 1.0)
        * (0.9 + 0.2 * np.minimum(s.lead, 90) / 90.0)
    )


def true_choice_probs(s: Sessions, price: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """P(customer books offer k) for every offer, shape (n,K)."""
    sens = true_sensitivity(s)[:, None]
    rel = price / s.ref
    cap_gap = np.minimum(0, C.CLASS_CAPACITY[s.cls] - s.party[:, None])
    u = 0.6 * (s.rating - 4.0) + 0.8 * cap_gap - 4.0 * sens * (rel - 1.0) - 0.5 * np.log1p(rank)
    u0 = 3.8 - 0.3 * s.returning + 0.2 * s.mobile
    e = np.exp(u)
    denom = np.exp(u0) + e.sum(axis=1)
    return e / denom[:, None]


def sample_choice(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Chosen offer index, or -1 for no booking."""
    cum = np.cumsum(probs, axis=1)
    r = rng.random(len(probs))[:, None]
    hit = r < cum
    chosen = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    return chosen


def net_factor(s: Sessions) -> np.ndarray:
    """Share of the margin CarTrawler keeps after the partner's revenue share."""
    return 1.0 - C.PARTNER_SHARE[s.partner]


def realised(s: Sessions, price: np.ndarray, chosen: np.ndarray):
    """Booked flag, CarTrawler net margin and gross booking value per session."""
    idx = np.arange(len(s))
    booked = chosen >= 0
    safe = np.where(booked, chosen, 0)
    gross = (price[idx, safe] - s.cost[idx, safe]) * s.trip
    gbv = price[idx, safe] * s.trip
    return booked, np.where(booked, gross * net_factor(s), 0.0), np.where(booked, gbv, 0.0)


def expected_outcomes(s: Sessions, price: np.ndarray, rank: np.ndarray):
    """Exact expected booking rate and CarTrawler net margin per session."""
    p = true_choice_probs(s, price, rank)
    margin = ((price - s.cost) * s.trip[:, None] * p).sum(axis=1) * net_factor(s)
    return p.sum(axis=1), margin


def expected_gbv(s: Sessions, price: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Exact expected gross booking value per session (what travellers spend)."""
    p = true_choice_probs(s, price, rank)
    return (price * s.trip[:, None] * p).sum(axis=1)


def partner_commission(s: Sessions, net_margin: np.ndarray) -> np.ndarray:
    """The partner's revenue share implied by CarTrawler's net margin."""
    share = C.PARTNER_SHARE[s.partner]
    return net_margin * share / (1.0 - share)


# ----------------------------------------------------------------------------
# Simple policies
# ----------------------------------------------------------------------------
def control_policy(s: Sessions):
    """Current-style baseline: flat 20% markup, cheapest first."""
    price = s.cost * (1 + C.BASELINE_MARKUP)
    return price, rank_from_key(price)


def logging_policy(s: Sessions, rng: np.random.Generator):
    """Exploratory policy used to collect training logs.

    Random markups and, half the time, a random order. This randomisation is what
    lets a model learn price and position effects instead of just echoing the
    old rules.
    """
    n, k = s.cost.shape
    markup = rng.uniform(C.MARKUP_GRID.min(), C.MARKUP_GRID.max(), (n, k))
    price = s.cost * (1 + markup)
    shuffled = rank_from_key(rng.random((n, k)))
    by_price = rank_from_key(price)
    rank = np.where((rng.random(n) < 0.5)[:, None], shuffled, by_price)
    return price, rank
