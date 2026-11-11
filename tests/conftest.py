import pytest

from ancillary.model import ConversionModel
from ancillary.train import make_dataset


@pytest.fixture(scope="session")
def small_model(tmp_path_factory):
    _s, X, y = make_dataset(20_000, seed=3)
    f = X.shape[-1]
    m = ConversionModel.fit(X.reshape(-1, f), y.reshape(-1), meta={"version": "test"})
    path = tmp_path_factory.mktemp("model") / "model.json"
    m.save(path)
    return m, path
