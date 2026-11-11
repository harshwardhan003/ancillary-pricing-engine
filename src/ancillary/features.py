"""Feature construction shared by training, the optimiser and the API.

Works on arrays of shape (..., K) for offer-level inputs and (..., 1) for
session-level inputs, so the optimiser can score many candidate price
arrangements in one call.
"""
from __future__ import annotations

import numpy as np

from . import config as C

_MK = [f"market_{m}" for m in C.MARKETS[1:]]
_CL = [f"class_{c}" for c in C.CAR_CLASSES[1:]]
_PT = [f"partner_{p}" for p in C.PARTNER_SLUG[1:]]
FEATURE_NAMES = (
    ["rel_price_ref", "rel_price_set", "ln_rank", "rating_c", "cap_gap",
     "returning", "mobile", "ln_lead", "trip_days", "party_size"]
    + _MK + _CL + _PT
    + ["rel_x_mobile", "rel_x_returning", "rel_x_ln_lead"]
    + [f"rel_x_{m}" for m in _MK]
    + [f"rel_x_{p}" for p in _PT]
)


def build_features(lead, trip, party, market, mobile, returning, partner,
                   cls, rating, price, ref, rank) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    shp = price.shape

    def b(a):
        return np.broadcast_to(np.asarray(a, dtype=float), shp)

    rel_ref = price / ref - 1.0
    rel_set = price / price.mean(axis=-1, keepdims=True) - 1.0
    cap_gap = np.minimum(0.0, C.CLASS_CAPACITY[cls] - b(party))
    ln_lead = np.log1p(b(lead))
    mobile_b, ret_b = b(mobile), b(returning)
    mk = [b(np.asarray(market) == m) for m in range(1, len(C.MARKETS))]
    cl = [b(np.asarray(cls) == c) for c in range(1, len(C.CAR_CLASSES))]
    pt = [b(np.asarray(partner) == k) for k in range(1, len(C.PARTNERS))]

    cols = (
        [rel_ref, rel_set, np.log1p(b(rank)), b(rating) - 4.0, cap_gap, ret_b,
         mobile_b, ln_lead, b(trip), b(party)]
        + mk + cl + pt
        + [rel_ref * mobile_b, rel_ref * ret_b, rel_ref * ln_lead]
        + [rel_ref * m for m in mk]
        + [rel_ref * q for q in pt]
    )
    return np.stack([np.broadcast_to(c, shp) for c in cols], axis=-1)


def features_from_sessions(s, price, rank) -> np.ndarray:
    return build_features(*s.ctx(), s.cls, s.rating, price, s.ref, rank)
