import subprocess
import sys
import textwrap

import keras
import numpy as np
import pytest
import torch

from mlpp_lib import models
from mlpp_lib.probabilistic_layers import DISTRIBUTIONS
from mlpp_lib.utils import get_loss, get_metric

N_INPUTS = 5

TEST_METRICS = [
    "expected_bias",
    "expected_mean_absolute_error",
    {"MAEBusts": {"threshold": 0.5}},
]

MODELS = {
    "fcn": lambda **kw: models.fully_connected_network(
        2, hidden_layers=[3], dropout=0.1, **kw
    ),
    "multibranch": lambda **kw: models.fully_connected_multibranch_network(
        2, hidden_layers=[3], **kw
    ),
    "dcn": lambda **kw: models.deep_cross_network(
        2, hidden_layers=[3], cross_layers_hiddensize=4, **kw
    ),
}


def _loss_for(distribution):
    if distribution == "MultivariateNormalTriL":
        return {"EnergyScore": {"num_samples": 10}}
    if distribution in ("Poisson", "Bernoulli", "MixtureNormal"):
        return "NegativeLogLikelihood"
    return {"CRPSEnsemble": {"num_samples": 10}}


def _assert_same_predictions(model, loaded_model, inputs):
    for w1, w2 in zip(model.weights, loaded_model.weights):
        np.testing.assert_allclose(w1.numpy(), w2.numpy())
    pred, loaded_pred = model(inputs), loaded_model(inputs)
    torch.testing.assert_close(pred.mean, loaded_pred.mean, equal_nan=True)
    torch.testing.assert_close(pred.variance, loaded_pred.variance, equal_nan=True)


@pytest.mark.parametrize("model_name", list(MODELS))
@pytest.mark.parametrize("distribution", list(DISTRIBUTIONS))
def test_save_model(distribution, model_name, tmp_path):
    """Test model save/load"""
    path = tmp_path / "model.keras"

    model = MODELS[model_name](
        probabilistic_layer=distribution,
        out_bias_init=np.array([0.1, 0.2]),
    )
    inputs = keras.random.normal((32, N_INPUTS))
    model(inputs)
    model.compile(
        loss=get_loss(_loss_for(distribution)),
        metrics=[get_metric(metric) for metric in TEST_METRICS],
    )
    model.save(path)

    loaded_model = keras.saving.load_model(path)
    assert isinstance(loaded_model, type(model))
    assert type(loaded_model.loss) is type(model.loss)
    _assert_same_predictions(model, loaded_model, inputs)
    assert (
        loaded_model.output_distribution.get_config()
        == model.output_distribution.get_config()
    )


def test_save_model_deterministic(tmp_path):
    path = tmp_path / "model.keras"
    model = models.fully_connected_network(2, hidden_layers=[3])
    inputs = keras.random.normal((32, N_INPUTS))
    model(inputs)
    model.compile(loss=get_loss("mse"), metrics=[get_metric("expected_bias")])
    model.save(path)
    loaded_model = keras.saving.load_model(path)
    np.testing.assert_allclose(
        model(inputs).detach().numpy(), loaded_model(inputs).detach().numpy()
    )


def test_load_model_in_new_process(tmp_path):
    """The model must be loadable in a fresh process, without unsafe deserialization."""
    path = tmp_path / "model.keras"
    inputs = np.random.default_rng(0).normal(size=(8, N_INPUTS)).astype("float32")
    model = models.fully_connected_network(
        2,
        hidden_layers=[3],
        probabilistic_layer="TruncatedNormal",
        prob_layer_kwargs={"low": -1.0},
    )
    model(inputs)
    model.compile(loss=get_loss("CRPSTruncatedNormal"))
    model.save(path)
    np.save(tmp_path / "inputs.npy", inputs)
    np.save(tmp_path / "mean.npy", model(inputs).mean.detach().numpy())

    script = textwrap.dedent(f"""
        import numpy as np
        import mlpp_lib  # registers the custom objects and sets the backend
        import keras

        model = keras.saving.load_model({str(path)!r}, safe_mode=True)
        mean = model(np.load({str(tmp_path / "inputs.npy")!r})).mean
        np.testing.assert_allclose(
            mean.detach().numpy(), np.load({str(tmp_path / "mean.npy")!r}), rtol=1e-6
        )
        assert model.output_distribution.distribution.low == -1.0
        """)
    subprocess.run([sys.executable, "-c", script], check=True)


def test_save_model_mlflow(tmp_path):
    """Test model save/load with mlflow"""
    mlflow = pytest.importorskip("mlflow")

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.absolute()}/mlflow.db")

    model = models.fully_connected_network(
        2, hidden_layers=[3], probabilistic_layer="Normal"
    )
    inputs = keras.random.normal((32, N_INPUTS))
    model(inputs)
    model.compile(loss=get_loss("CRPSNormal"))

    with mlflow.start_run():
        model_info = mlflow.keras.log_model(model, name="model")

    loaded_model = mlflow.keras.load_model(model_info.model_uri)
    assert isinstance(loaded_model, models.ProbabilisticModel)
    _assert_same_predictions(model, loaded_model, inputs)
