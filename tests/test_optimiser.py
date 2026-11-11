import numpy as np

from ancillary import config as C
from ancillary.optimiser import optimise
from ancillary.simulator import control_policy, expected_outcomes, sample_sessions


def test_plan_is_valid(small_model):
    m, _ = small_model
    s = sample_sessions(300, np.random.default_rng(5))
    plan = optimise(m, s, lam=5.0)
    assert np.isin(plan.markup, C.MARKUP_GRID).all()
    assert np.allclose(plan.price, s.cost * (1 + plan.markup))
    # ranks are a permutation of 0..K-1 in every session
    assert (np.sort(plan.rank, axis=1) == np.arange(C.N_OFFERS)).all()


def test_optimiser_improves_expected_margin_under_true_model(small_model):
    m, _ = small_model
    s = sample_sessions(8_000, np.random.default_rng(6))
    _c_conv, c_margin = expected_outcomes(s, *control_policy(s))
    plan = optimise(m, s, lam=0.0)
    _, o_margin = expected_outcomes(s, plan.price, plan.rank)
    assert o_margin.mean() > c_margin.mean()


def test_higher_booking_value_pushes_prices_down_and_conversion_up(small_model):
    m, _ = small_model
    s = sample_sessions(4_000, np.random.default_rng(7))
    lo, hi = optimise(m, s, lam=0.0), optimise(m, s, lam=40.0)
    assert hi.markup.mean() < lo.markup.mean()
    assert expected_outcomes(s, hi.price, hi.rank)[0].mean() > expected_outcomes(s, lo.price, lo.rank)[0].mean()


def test_single_session_matches_batch(small_model):
    m, _ = small_model
    s = sample_sessions(10, np.random.default_rng(8))
    batch = optimise(m, s)
    one = optimise(m, s.take(slice(3, 4)))
    assert np.allclose(batch.price[3], one.price[0])
    assert (batch.rank[3] == one.rank[0]).all()


def test_partner_markup_limits_are_respected(small_model):
    m, _ = small_model
    s = sample_sessions(1_500, np.random.default_rng(9))
    plan = optimise(m, s, lam=0.0)
    lo, hi = C.PARTNER_MARKUP_MIN[s.partner], C.PARTNER_MARKUP_MAX[s.partner]
    assert (plan.markup >= lo[:, None] - 1e-9).all() and (plan.markup <= hi[:, None] + 1e-9).all()


def test_per_partner_lam_vector_matches_scalar_when_equal(small_model):
    m, _ = small_model
    s = sample_sessions(200, np.random.default_rng(10))
    a, b = optimise(m, s, lam=7.0), optimise(m, s, lam=np.full(len(C.PARTNERS), 7.0))
    assert np.allclose(a.price, b.price) and (a.rank == b.rank).all()
