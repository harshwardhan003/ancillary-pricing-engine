import numpy as np

from ancillary import config as C
from ancillary.simulator import (
    control_policy,
    expected_outcomes,
    rank_from_key,
    realised,
    sample_choice,
    sample_sessions,
    true_choice_probs,
)


def test_shapes_and_ranges():
    s = sample_sessions(500, np.random.default_rng(0))
    assert s.cost.shape == (500, C.N_OFFERS)
    assert (s.cost > 0).all() and (s.trip >= 1).all()
    # offers within a session are distinct (supplier, class) combos
    combo = s.sup * 10 + s.cls
    assert all(len(set(r)) == C.N_OFFERS for r in combo)


def test_choice_probs_are_valid_and_conversion_is_plausible():
    rng = np.random.default_rng(1)
    s = sample_sessions(20_000, rng)
    price, rank = control_policy(s)
    p = true_choice_probs(s, price, rank)
    assert (p >= 0).all() and (p.sum(axis=1) < 1).all()
    assert 0.03 < p.sum(axis=1).mean() < 0.12


def test_higher_price_lowers_booking_probability():
    s = sample_sessions(5_000, np.random.default_rng(2))
    price, rank = control_policy(s)
    lo = true_choice_probs(s, price, rank).sum(axis=1).mean()
    hi = true_choice_probs(s, price * 1.2, rank).sum(axis=1).mean()
    assert hi < lo


def test_top_position_converts_better():
    s = sample_sessions(5_000, np.random.default_rng(3))
    price, _ = control_policy(s)
    asc = rank_from_key(price)
    p = true_choice_probs(s, price, asc)
    top = p[asc == 0].mean()
    bottom = p[asc == C.N_OFFERS - 1].mean()
    assert top > bottom


def test_sampled_conversion_matches_expected():
    rng = np.random.default_rng(4)
    s = sample_sessions(100_000, rng)
    price, rank = control_policy(s)
    chosen = sample_choice(true_choice_probs(s, price, rank), rng)
    booked, margin, _ = realised(s, price, chosen)
    exp_conv, exp_margin = expected_outcomes(s, price, rank)
    assert abs(booked.mean() - exp_conv.mean()) < 0.003
    assert abs(margin.mean() / exp_margin.mean() - 1) < 0.06
