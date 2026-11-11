import numpy as np

from ancillary import config as C
from ancillary.features import FEATURE_NAMES
from ancillary.simulator import control_policy, expected_outcomes, sample_sessions

s = sample_sessions(50_000, np.random.default_rng(0))
print("features:", len(FEATURE_NAMES), "(expect 41)")
print("partner share of sessions:", np.round(np.bincount(s.partner) / len(s), 3))
price, rank = control_policy(s)
conv, margin = expected_outcomes(s, price, rank)
for i, name in enumerate(C.PARTNERS):
    k = s.partner == i
    print(f"{name:20s} booking rate {conv[k].mean():.4f}  net margin/session {margin[k].mean():.2f}")
print("overall booking rate:", round(float(conv.mean()), 4))
