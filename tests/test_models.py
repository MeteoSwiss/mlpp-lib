import itertools

import numpy as np
import pytest
import tensorflow as tf
from keras.engine.functional import Functional
from numpy.testing import assert_array_equal

from mlpp_lib import models


FCN_OPTIONS = dict(
    input_shape=[(5,)],
    output_size=[1, 2],
    hidden_layers=[[8, 8]],
    activations=["relu", ["relu", "elu"]],
    dropout=[None, 0.1, [0.1, 0.0]],
    mc_dropout=[True, False],
    out_bias_init=["zeros", np.array([0.2]), np.array([0.2, 2.1])],
    probabilistic_layer=[None, "IndependentNormal", "MultivariateNormalTriL"],
    skip_connection=[False, True],
)


FCN_SCENARIOS = [
    dict(zip(list(FCN_OPTIONS.keys()), x))
    for x in itertools.product(*FCN_OPTIONS.values())
]


DCN_SCENARIOS = [
    dict(zip(list(FCN_OPTIONS.keys()), x))
    for x in itertools.product(*FCN_OPTIONS.values())
]


def _test_prediction(model, scenario_kwargs, dummy_input, output_size):
    pred = model(dummy_input)
    assert pred.shape == (32, output_size)
    pred2 = model(dummy_input)
    if scenario_kwargs["probabilistic_layer"] is not None:
        pred = pred.mean()
        pred2 = pred2.mean()

    if scenario_kwargs["dropout"] is not None and scenario_kwargs["mc_dropout"]:
        with pytest.raises(AssertionError):
            assert_array_equal(pred, pred2)
    else:
        assert_array_equal(pred, pred2)


@pytest.mark.parametrize("scenario_kwargs", FCN_SCENARIOS)
def test_fully_connected_network(scenario_kwargs):

    tf.keras.backend.clear_session()

    scenario_kwargs = scenario_kwargs.copy()
    input_shape = scenario_kwargs.pop("input_shape")
    output_size = scenario_kwargs.pop("output_size")
    dummy_input = np.random.randn(32, *input_shape)

    # check that correct errors are raised for some scenarios
    if isinstance(scenario_kwargs["out_bias_init"], np.ndarray):
        if scenario_kwargs["out_bias_init"].shape[-1] != output_size:
            with pytest.raises(AssertionError):
                models.fully_connected_network(
                    input_shape, output_size, **scenario_kwargs
                )
            return
        else:
            model = models.fully_connected_network(
                input_shape, output_size, **scenario_kwargs
            )

    else:
        model = models.fully_connected_network(
            input_shape, output_size, **scenario_kwargs
        )
        assert isinstance(model, Functional)

    _test_prediction(model, scenario_kwargs, dummy_input, output_size)


@pytest.mark.parametrize("scenario_kwargs", FCN_SCENARIOS)
def test_fully_connected_multibranch_network(scenario_kwargs):

    tf.keras.backend.clear_session()

    scenario_kwargs = scenario_kwargs.copy()
    input_shape = scenario_kwargs.pop("input_shape")
    output_size = scenario_kwargs.pop("output_size")
    dummy_input = np.random.randn(32, *input_shape)

    # check that correct errors are raised for some scenarios
    if isinstance(scenario_kwargs["out_bias_init"], np.ndarray):
        if scenario_kwargs["out_bias_init"].shape[-1] != output_size:
            with pytest.raises(AssertionError):
                models.fully_connected_multibranch_network(
                    input_shape, output_size, **scenario_kwargs
                )
            return
        else:
            model = models.fully_connected_multibranch_network(
                input_shape, output_size, **scenario_kwargs
            )

    else:
        model = models.fully_connected_multibranch_network(
            input_shape, output_size, **scenario_kwargs
        )
        assert isinstance(model, Functional)

    _test_prediction(model, scenario_kwargs, dummy_input, output_size)


@pytest.mark.parametrize("scenario_kwargs", DCN_SCENARIOS)
def test_deep_cross_network(scenario_kwargs):

    scenario_kwargs = scenario_kwargs.copy()
    input_shape = scenario_kwargs.pop("input_shape")
    output_size = scenario_kwargs.pop("output_size")
    dummy_input = np.random.randn(32, *input_shape)
    # check that correct errors are raised for some scenarios
    if isinstance(scenario_kwargs["out_bias_init"], np.ndarray):
        if scenario_kwargs["out_bias_init"].shape[-1] != output_size:
            with pytest.raises(AssertionError):
                models.deep_cross_network(input_shape, output_size, **scenario_kwargs)
            return
        else:
            model = models.deep_cross_network(
                input_shape, output_size, **scenario_kwargs
            )

    else:
        model = models.deep_cross_network(input_shape, output_size, **scenario_kwargs)
        assert isinstance(model, Functional)

    _test_prediction(model, scenario_kwargs, dummy_input, output_size)
