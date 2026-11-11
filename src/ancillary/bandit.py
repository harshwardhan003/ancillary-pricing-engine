"""Online learning with Thompson sampling.

A complementary approach to the offline model: learn the best flat markup per
segment (partner x device) directly from realised net margin, with no model at
all. This is the cold-start tool for onboarding a new partner, before there are
enough logs to train on, and it also serves as a benchmark.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .baselines import N_SEG, allowed_by_segment, flat_markup_plan, segment, segment_markup_table
from .simulator import (
    TrafficMix,
    expected_outcomes,
    realised,
    sample_choice,
    sample_sessions,
    true_choice_probs,
)


class ThompsonMarkup:
    def __init__(self, n_segments: int, grid: np.ndarray, rng: np.random.Generator,
                 prior_var: float = 400.0, allowed: np.ndarray | None = None):
        self.grid, self.rng, self.prior_var = grid, rng, prior_var
        self.allowed = allowed if allowed is not None else np.ones((n_segments, len(grid)), bool)
        shape = (n_segments, len(grid))
        self.n = np.zeros(shape)
        self.s1 = np.zeros(shape)
        self.s2 = np.zeros(shape)

    def choose(self, seg: np.ndarray) -> np.ndarray:
        n = np.maximum(self.n[seg], 1e-9)
        mean = np.where(self.n[seg] > 0, self.s1[seg] / n, 0.0)
        var = np.where(self.n[seg] > 1, self.s2[seg] / n - mean**2, self.prior_var)
        sd = np.sqrt(np.maximum(var, 1e-6) / (self.n[seg] + 1))
        draw = mean + sd * self.rng.standard_normal(mean.shape)
        return np.where(self.allowed[seg], draw, -np.inf).argmax(axis=1)

    def update(self, seg, arm, reward):
        np.add.at(self.n, (seg, arm), 1)
        np.add.at(self.s1, (seg, arm), reward)
        np.add.at(self.s2, (seg, arm), reward**2)


def simulate_bandit(total: int = 200_000, batch: int = 1_000, seed: int = 0,
                    mix: TrafficMix | None = None) -> dict:
    """Pass mix=TrafficMix.only(i) to simulate onboarding a single new partner."""
    rng = np.random.default_rng(seed)
    grid = C.MARKUP_GRID
    allowed = allowed_by_segment(grid)
    bandit = ThompsonMarkup(N_SEG, grid, rng, allowed=allowed)
    rewards, base_rewards, _exp_gap = [], [], []

    for _ in range(total // batch):
        s = sample_sessions(batch, rng, mix)
        seg = segment(s)
        arm = bandit.choose(seg)
        price, rank = flat_markup_plan(s, grid[arm])
        chosen = sample_choice(true_choice_probs(s, price, rank), rng)
        _, margin, _ = realised(s, price, chosen)
        bandit.update(seg, arm, margin)
        rewards.append(expected_outcomes(s, price, rank)[1].mean())
        bp, br = flat_markup_plan(s, np.full(batch, C.BASELINE_MARKUP))
        base_rewards.append(expected_outcomes(s, bp, br)[1].mean())

    rewards, base_rewards = np.array(rewards), np.array(base_rewards)
    tail = slice(int(len(rewards) * 0.8), None)

    # Best single markup per segment in hindsight (exact expectation on a big sample).
    s = sample_sessions(60_000, np.random.default_rng(seed + 1), mix)
    table, weights = segment_markup_table(s, grid)
    hindsight = float((np.where(allowed, table, -np.inf).max(axis=1) * weights).sum())
    baseline = float((table[:, int(np.argmin(abs(grid - C.BASELINE_MARKUP)))] * weights).sum())

    return {
        "sessions": total,
        "baseline_margin_per_session": baseline,
        "hindsight_best_margin_per_session": hindsight,
        "bandit_early_margin_per_session": float(rewards[:20].mean()),
        "bandit_final_margin_per_session": float(rewards[tail].mean()),
        "final_uplift_vs_baseline": float(rewards[tail].mean() / base_rewards[tail].mean() - 1),
        "share_of_hindsight_gain_captured": float(
            (rewards[tail].mean() - base_rewards[tail].mean()) / (hindsight - baseline)
        ),
        "learning_curve": [float(x) for x in rewards.reshape(-1, 10).mean(axis=1)]
        if len(rewards) % 10 == 0 else [float(x) for x in rewards],
    }
