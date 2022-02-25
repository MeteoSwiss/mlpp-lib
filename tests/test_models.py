import itertools

import numpy as np
from keras.engine.functional import Functional
import pytest

from mlpp_lib import models

FCN_OPTIONS = dict(
    input_shape=[(5,)],
    output_size=[1, 2],
    hidden_layers=[[8, 8]],
    activations=["relu", ["relu", "elu"]],
    dropout=[None, 0.1, [0.1, 0.0]],
    out_bias_init=["zeros", np.array([0.2]), np.array([0.2, 2.1])],
    probabilistic_layer=[None, "IndependentNormal", "MultivariateNormalTriL"],
)

FCN_SCENARIOS = [
    dict(zip(list(FCN_OPTIONS.keys()), x))
    for x in itertools.product(*FCN_OPTIONS.values())
]


@pytest.mark.parametrize("scenario_kwargs", FCN_SCENARIOS)
def test_fully_connected_network(scenario_kwargs):

    input_shape = scenario_kwargs.pop("input_shape")
    output_size = scenario_kwargs.pop("output_size")

    # check that correct errors are raised for some scenarios
    if isinstance(scenario_kwargs["out_bias_init"], np.ndarray):
        if scenario_kwargs["out_bias_init"].shape[-1] != output_size:
            with pytest.raises(AssertionError):
                models.fully_connected_network(
                    input_shape, output_size, **scenario_kwargs
                )
            pass

    else:
        model = models.fully_connected_network(
            input_shape, output_size, **scenario_kwargs
        )
        assert isinstance(model, Functional)
