import numpy as np

from ancillary.features import FEATURE_NAMES, features_from_sessions
from ancillary.model import ConversionModel
from ancillary.simulator import control_policy, sample_sessions


def test_feature_shapes_and_broadcasting():
    s = sample_sessions(50, np.random.default_rng(0))
    price, rank = control_policy(s)
    X = features_from_sessions(s, price, rank)
    assert X.shape == (50, 6, len(FEATURE_NAMES))
    # a candidate axis in front must give the same numbers for the same inputs
    from ancillary.features import build_features
    Xg = build_features(*s.ctx(), s.cls, s.rating, np.broadcast_to(price, (3, 50, 6)), s.ref, rank)
    assert Xg.shape == (3, 50, 6, len(FEATURE_NAMES))
    assert np.allclose(Xg[1], X)


def test_model_learns_sensible_signs(small_model):
    m, _ = small_model
    w = dict(zip(FEATURE_NAMES, m.w))
    assert w["rel_price_ref"] < 0       # higher price, lower conversion
    assert w["ln_rank"] < 0             # lower position, lower conversion
    assert w["rating_c"] > 0            # better supplier, higher conversion
    assert w["cap_gap"] > 0             # car too small for the party hurts


def test_predictions_are_probabilities(small_model):
    m, _ = small_model
    s = sample_sessions(200, np.random.default_rng(1))
    p = m.predict(features_from_sessions(s, *control_policy(s)))
    assert ((p > 0) & (p < 1)).all()


def test_save_load_roundtrip(small_model, tmp_path):
    m, _ = small_model
    path = tmp_path / "m.json"
    m.save(path)
    m2 = ConversionModel.load(path)
    s = sample_sessions(50, np.random.default_rng(2))
    X = features_from_sessions(s, *control_policy(s))
    assert np.allclose(m.predict(X), m2.predict(X))


def test_mlp_challenger_roundtrip_and_valid_probabilities(tmp_path):
    import numpy as np

    from ancillary.mlp import MLPModel
    from ancillary.train import make_dataset

    _s, X, y = make_dataset(6_000, seed=4)
    f = X.shape[-1]
    m = MLPModel.fit(X.reshape(-1, f), y.reshape(-1), hidden=(8,), max_epochs=3)
    m.save(tmp_path / "mlp.json")
    m2 = MLPModel.load(tmp_path / "mlp.json")
    p = m.predict(X.reshape(-1, f)[:500])
    assert ((p > 0) & (p < 1)).all()
    assert np.allclose(p, m2.predict(X.reshape(-1, f)[:500]))
