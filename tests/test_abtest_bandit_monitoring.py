import numpy as np

from ancillary import config as C
from ancillary.abtest import run_ab
from ancillary.bandit import ThompsonMarkup, simulate_bandit
from ancillary.monitoring import DriftMonitor, psi
from ancillary.simulator import TrafficMix, control_policy, sample_sessions


def test_aa_test_finds_no_difference():
    s = sample_sessions(60_000, np.random.default_rng(10))
    res = run_ab(s, control_policy, control_policy, np.random.default_rng(11))
    lo, hi = res["margin_uplift"]["ci95"]
    assert lo < 0 < hi
    assert res["margin_uplift"]["p_value"] > 0.01
    assert abs(res["analytic"]["margin_uplift"]) < 1e-9


def test_ab_detects_a_real_improvement():
    s = sample_sessions(80_000, np.random.default_rng(12))

    def better(sess):
        price = sess.cost * 1.32
        return price, np.argsort(np.argsort(-price, axis=1), axis=1)

    res = run_ab(s, control_policy, better, np.random.default_rng(13))
    assert res["analytic"]["margin_uplift"] > 0
    assert res["margin_uplift"]["relative"] > 0


def test_psi_basics():
    assert psi(np.array([.5, .5]), np.array([.5, .5])) < 1e-9
    assert psi(np.array([.5, .5]), np.array([.9, .1])) > 0.25


def test_drift_monitor_flags_shift_but_not_same_distribution():
    rng = np.random.default_rng(20)
    mon = DriftMonitor.fit(sample_sessions(30_000, rng))
    same = mon.check(sample_sessions(10_000, rng))
    shifted = mon.check(sample_sessions(10_000, rng, TrafficMix(lead_mult=0.3, mobile_add=0.25, partner_probs=(0.05, 0.05, 0.05, 0.05, 0.4, 0.4))))
    assert same["_overall"] == "ok"
    assert shifted["_overall"] == "alert"
    assert shifted["lead_days"]["status"] == "alert"


def test_bandit_prefers_better_arm():
    rng = np.random.default_rng(30)
    b = ThompsonMarkup(1, C.MARKUP_GRID, rng)
    best = 6
    for _ in range(400):
        arm = b.choose(np.zeros(50, dtype=int))
        reward = 1.0 - 0.05 * abs(arm - best) + rng.normal(0, 0.3, 50)
        b.update(np.zeros(50, dtype=int), arm, reward)
    final = np.bincount(b.choose(np.zeros(500, dtype=int)), minlength=10).argmax()
    assert abs(final - best) <= 1


def test_bandit_simulation_beats_baseline():
    res = simulate_bandit(total=40_000, batch=1_000, seed=1)
    assert res["final_uplift_vs_baseline"] > 0
