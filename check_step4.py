import numpy as np

from ancillary import config as C
from ancillary.model import ConversionModel
from ancillary.optimiser import optimise
from ancillary.simulator import control_policy, expected_outcomes, sample_sessions

model = ConversionModel.load("artifacts/model.json")
s = sample_sessions(8_000, np.random.default_rng(6))
cc, cm = expected_outcomes(s, *control_policy(s))
print(f"control: booking rate {cc.mean():.4f}  net margin/session {cm.mean():.3f}")

for lam in (0, 10, 30):
    plan = optimise(model, s, lam=float(lam))
    c, m = expected_outcomes(s, plan.price, plan.rank)
    print(f"lam={lam:>2}: bookings {c.mean() / cc.mean() - 1:+.1%}  net margin {m.mean() / cm.mean() - 1:+.1%}  mean markup {plan.markup.mean():.3f}")

lo, hi = C.PARTNER_MARKUP_MIN[s.partner], C.PARTNER_MARKUP_MAX[s.partner]
ok = (plan.markup >= lo[:, None] - 1e-9).all() and (plan.markup <= hi[:, None] + 1e-9).all()
print("all markups inside each partner's limits:", ok)
