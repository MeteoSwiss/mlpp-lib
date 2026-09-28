import itertools
import warnings

import keras
import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from mlpp_lib import models
from mlpp_lib.probabilistic_layers import DISTRIBUTIONS, WrappingTorchDist
from mlpp_lib.utils import get_model

N_INPUTS, BATCH = 5, 32

OPTIONS = dict(
    output_size=[1, 2],
    hidden_layers=[[8, 8]],
    activations=["relu", ["relu", "elu"]],
    dropout=[None, 0.1, [0.1, 0.0]],
    mc_dropout=[True, False],
    out_bias_init=["zeros", np.array([0.2]), np.array([0.2, 2.1])],
    probabilistic_layer=[None, "Gamma", "MultivariateNormalTriL"],
    skip_connection=[False, True],
)

SCENARIOS = [
    dict(zip(OPTIONS.keys(), values)) for values in itertools.product(*OPTIONS.values())
]


def _scenario_id(scenario):
    return "-".join(
        f"{k}={v.tolist() if isinstance(v, np.ndarray) else v}"
        for k, v in scenario.items()
    )


def _parameters(output):
    """Parameters of the predicted distribution (or the prediction itself)."""
    if isinstance(output, WrappingTorchDist):
        return [output.mean, output.variance]
    return [output]


def _check_model(model_fn, scenario):
    scenario = scenario.copy()
    output_size = scenario.pop("output_size")
    out_bias_init = scenario["out_bias_init"]
    if isinstance(out_bias_init, np.ndarray) and len(out_bias_init) != output_size:
        with pytest.raises(ValueError, match="Bias initialization"):
            model_fn(output_size, **scenario)
        return

    model = model_fn(output_size, **scenario)
    assert isinstance(model, keras.Model)
    inputs = torch.randn(BATCH, N_INPUTS)
    pred1 = model(inputs)
    pred2 = model(inputs)

    if scenario["probabilistic_layer"] is None:
        assert pred1.shape == (BATCH, output_size)
    else:
        assert isinstance(pred1, WrappingTorchDist)
        assert pred1.mean.shape == (BATCH, output_size)
        assert model(inputs, output_type="expected").shape == (BATCH, output_size)
        samples = model(inputs, output_type="samples", num_samples=3)
        assert samples.shape == (3, BATCH, output_size)

    # MC dropout makes predictions stochastic, standard dropout does not at inference
    is_deterministic = scenario["dropout"] is None or not scenario["mc_dropout"]
    for p1, p2 in zip(_parameters(pred1), _parameters(pred2)):
        p1, p2 = p1.detach().numpy(), p2.detach().numpy()
        if is_deterministic:
            assert_array_equal(p1, p2)
        else:
            assert not np.array_equal(p1, p2)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_fully_connected_network(scenario):
    _check_model(models.fully_connected_network, scenario)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_fully_connected_multibranch_network(scenario):
    _check_model(models.fully_connected_multibranch_network, scenario)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_deep_cross_network(scenario):
    _check_model(models.deep_cross_network, scenario)


@pytest.mark.parametrize("distribution", list(DISTRIBUTIONS))
@pytest.mark.parametrize(
    "model_fn",
    [
        models.fully_connected_network,
        models.fully_connected_multibranch_network,
        models.deep_cross_network,
    ],
)
def test_all_distributions(model_fn, distribution):
    model = model_fn(3, hidden_layers=[8], probabilistic_layer=distribution)
    output = model(torch.randn(BATCH, N_INPUTS))
    assert output.mean.shape == (BATCH, 3)


def test_out_bias_init():
    bias = np.array([0.5, -2.0])
    model = models.fully_connected_network(
        2, hidden_layers=[4], probabilistic_layer="Normal", out_bias_init=bias
    )
    model(torch.randn(BATCH, N_INPUTS))
    encoder_bias = model.output_distribution.parameters_encoder.bias.numpy()
    assert_array_equal(encoder_bias, [0.5, -2.0, 0.0, 0.0])

    model = models.fully_connected_network(2, hidden_layers=[4], out_bias_init=bias)
    model(torch.randn(BATCH, N_INPUTS))
    assert_array_equal(model.layers[-1].bias.numpy(), bias)


def test_dropout_float_is_not_applied_to_last_hidden_layer():
    model = models.fully_connected_network(1, hidden_layers=[8, 8, 8], dropout=0.1)
    model(torch.randn(BATCH, N_INPUTS))
    mlp = model.layers[0]
    assert sum(isinstance(l, keras.layers.Dropout) for l in mlp.layers) == 2


def test_multibranch_number_of_branches():
    # by default, one branch per distribution parameter
    model = models.fully_connected_multibranch_network(
        2, hidden_layers=[4], probabilistic_layer="Normal"
    )
    assert len(model.encoder.branches) == 4
    model = models.fully_connected_multibranch_network(2, hidden_layers=[4])
    assert len(model.layers[0].branches) == 2
    model = models.fully_connected_multibranch_network(
        2, hidden_layers=[4], n_branches=3, aggregation="sum"
    )
    assert len(model.layers[0].branches) == 3
    assert model(torch.randn(BATCH, N_INPUTS)).shape == (BATCH, 2)


def test_legacy_probabilistic_layer_config():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = models.fully_connected_network(
            2,
            hidden_layers=[4],
            probabilistic_layer={"IndependentTruncatedNormal": {"event_shape": [2]}},
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in record)
    distribution = model.output_distribution.distribution
    assert distribution.name == "TruncatedNormal"
    assert (distribution.low, distribution.high) == (0.0, float("inf"))

    model = models.fully_connected_network(
        2,
        hidden_layers=[4],
        probabilistic_layer={"TruncatedNormal": {"low": -1.0}},
        prob_layer_kwargs={"high": 1.0},
    )
    distribution = model.output_distribution.distribution
    assert (distribution.low, distribution.high) == (-1.0, 1.0)


def test_get_model_builds_the_model():
    model = get_model(
        (N_INPUTS,),
        (2,),
        {
            "fully_connected_network": {
                "hidden_layers": [4],
                "probabilistic_layer": "Normal",
            }
        },
    )
    assert model.built
    # (4 + 1) * 5 + (2 * 2) * (4 + 1)
    assert model.count_params() == 4 * N_INPUTS + 4 + 4 * 4 + 4
