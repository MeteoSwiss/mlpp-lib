import torch
from inspect import getmembers, isclass
import pytest

from mlpp_lib.layers import FullyConnectedLayer
from mlpp_lib.probabilistic_layers import (
    BaseDistributionLayer, 
    BaseParametricDistribution, 
    UniveriateGaussianDistribution,
    distribution_to_layer
)
from mlpp_lib.models import (
    fully_connected_network, 
    fully_connected_multibranch_network,
    deep_cross_network,
)

from mlpp_lib import probabilistic_layers

DISTRIBUTIONS = [obj[1] for obj in getmembers(probabilistic_layers, isclass) 
                 if issubclass(obj[1], BaseParametricDistribution) and obj[0] != 'BaseParametricDistribution']


@pytest.mark.parametrize("distribution", list(distribution_to_layer.keys())+[None])
@pytest.mark.parametrize("skip_connection", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_fcn_model_creation(distribution, skip_connection, batchnorm):
    
    hidden_layers=[16,16,8]
    output_size = 4
    
    model = fully_connected_network(output_size=output_size, 
                            hidden_layers=hidden_layers,
                            batchnorm=batchnorm,
                            skip_connection=skip_connection,
                            probabilistic_layer=distribution)
    
     
    inputs = torch.randn(32,6)
    
    output = model(inputs)
    
    
@pytest.mark.parametrize("distribution", list(distribution_to_layer.keys())+[None])
@pytest.mark.parametrize("skip_connection", [True, False])
@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("aggregation", ['sum', 'concat'])
def test_multibranch_fcn_creation(distribution, skip_connection, batchnorm, aggregation):
    hidden_layers=[16,16,8]
    output_size = 4
    n_branches = 3
    
    model = fully_connected_multibranch_network(output_size=output_size, 
                                                hidden_layers=hidden_layers,
                                                batchnorm=batchnorm,
                                                skip_connection=skip_connection,
                                                probabilistic_layer=distribution,
                                                n_branches=n_branches,
                                                aggregation=aggregation)
    
    
    inputs = torch.randn(32,6)
    
    output = model(inputs)
    

@pytest.mark.parametrize("distribution", list(distribution_to_layer.keys())+[None], ids=lambda d: f"distribution={d}" if d else "distribution=None")
def test_deep_cross_network(distribution):
    hidden_layers=[16,16,8]
    output_size = 4
    n_crosses = 3
    
    model = deep_cross_network(output_size=output_size,
                       hidden_layers=hidden_layers,
                       n_cross_layers=n_crosses,
                       cross_layers_hiddensize=16,
                       probabilistic_layer=distribution)
    
    inputs = torch.randn(32,6)
    
    output = model(inputs)
    

# import itertools

# import numpy as np
# import pytest
# import tensorflow as tf
# from tensorflow.keras import Model
# from numpy.testing import assert_array_equal

# from mlpp_lib import models


# FCN_OPTIONS = dict(
#     input_shape=[(5,)],
#     output_size=[1, 2],
#     hidden_layers=[[8, 8]],
#     activations=["relu", ["relu", "elu"]],
#     dropout=[None, 0.1, [0.1, 0.0]],
#     mc_dropout=[True, False],
#     out_bias_init=["zeros", np.array([0.2]), np.array([0.2, 2.1])],
#     probabilistic_layer=[None] + ["IndependentGamma", "MultivariateNormalDiag"],
#     skip_connection=[False, True],
# )


# FCN_SCENARIOS = [
#     dict(zip(list(FCN_OPTIONS.keys()), x))
#     for x in itertools.product(*FCN_OPTIONS.values())
# ]


# DCN_SCENARIOS = [
#     dict(zip(list(FCN_OPTIONS.keys()), x))
#     for x in itertools.product(*FCN_OPTIONS.values())
# ]


# def _test_model(model):
#     moodel_is_keras = isinstance(model, tf.keras.Model)
#     assert moodel_is_keras
#     assert isinstance(model, Model)
#     assert len(model.layers[-1]._inbound_nodes) > 0
#     model_output = model.layers[-1].output
#     assert not isinstance(
#         model_output, list
#     ), "The model output must be a single tensor!"
#     assert (
#         len(model_output.shape) < 3
#     ), "The model output must be a vector or a single value!"


# def _test_prediction(model, scenario_kwargs, dummy_input, output_size):
#     is_deterministic = (
#         scenario_kwargs["dropout"] is None or not scenario_kwargs["mc_dropout"]
#     )
#     is_probabilistic = scenario_kwargs["probabilistic_layer"] is not None
#     if is_probabilistic:
#         return

#     pred = model(dummy_input)
#     assert pred.shape == (32, output_size)
#     pred2 = model(dummy_input)

#     if is_deterministic:
#         assert_array_equal(pred, pred2)
#     else:
#         with pytest.raises(AssertionError):
#             assert_array_equal(pred, pred2)


# def _test_prediction_prob(model, scenario_kwargs, dummy_input, output_size):
#     is_deterministic = (
#         scenario_kwargs["dropout"] is None or not scenario_kwargs["mc_dropout"]
#     )
#     is_probabilistic = scenario_kwargs["probabilistic_layer"] is not None
#     if not is_probabilistic:
#         return

#     pred1 = model(dummy_input)
#     assert pred1.shape == (32, output_size)
#     pred2 = model(dummy_input)
#     try:
#         # Idependent layers have a "distribution" attribute
#         pred1_params = pred1.parameters["distribution"].parameters
#         pred2_params = pred2.parameters["distribution"].parameters
#     except KeyError:
#         pred1_params = pred1.parameters
#         pred2_params = pred2.parameters

#     for param in pred1_params.keys():
#         try:
#             param_array1 = pred1_params[param].numpy()
#             param_array2 = pred2_params[param].numpy()
#         except AttributeError:
#             continue

#         if is_deterministic:
#             assert_array_equal(param_array1, param_array2)
#         else:
#             with pytest.raises(AssertionError):
#                 assert_array_equal(param_array1, param_array2)


# @pytest.mark.parametrize("scenario_kwargs", FCN_SCENARIOS)
# def test_fully_connected_network(scenario_kwargs):

#     tf.keras.backend.clear_session()

#     scenario_kwargs = scenario_kwargs.copy()
#     input_shape = scenario_kwargs.pop("input_shape")
#     output_size = scenario_kwargs.pop("output_size")
#     dummy_input = np.random.randn(32, *input_shape)

#     # check that correct errors are raised for some scenarios
#     if isinstance(scenario_kwargs["out_bias_init"], np.ndarray):
#         if scenario_kwargs["out_bias_init"].shape[-1] != output_size:
#             with pytest.raises(AssertionError):
#                 models.fully_connected_network(
#                     input_shape, output_size, **scenario_kwargs
#                 )
#             return
#         else:
#             model = models.fully_connected_network(
#                 input_shape, output_size, **scenario_kwargs
#             )

#     else:
#         model = models.fully_connected_network(
#             input_shape, output_size, **scenario_kwargs
#         )

#     _test_model(model)
#     _test_prediction(model, scenario_kwargs, dummy_input, output_size)
#     _test_prediction_prob(model, scenario_kwargs, dummy_input, output_size)


# @pytest.mark.parametrize("scenario_kwargs", FCN_SCENARIOS)
# def test_fully_connected_multibranch_network(scenario_kwargs):

#     tf.keras.backend.clear_session()

#     scenario_kwargs = scenario_kwargs.copy()
#     input_shape = scenario_kwargs.pop("input_shape")
#     output_size = scenario_kwargs.pop("output_size")
#     dummy_input = np.random.randn(32, *input_shape)

#     # check that correct errors are raised for some scenarios
#     if isinstance(scenario_kwargs["out_bias_init"], np.ndarray):
#         if scenario_kwargs["out_bias_init"].shape[-1] != output_size:
#             with pytest.raises(AssertionError):
#                 models.fully_connected_multibranch_network(
#                     input_shape, output_size, **scenario_kwargs
#                 )
#             return
#         else:
#             model = models.fully_connected_multibranch_network(
#                 input_shape, output_size, **scenario_kwargs
#             )

#     else:
#         model = models.fully_connected_multibranch_network(
#             input_shape, output_size, **scenario_kwargs
#         )

#     _test_model(model)
#     _test_prediction(model, scenario_kwargs, dummy_input, output_size)
#     _test_prediction_prob(model, scenario_kwargs, dummy_input, output_size)


# @pytest.mark.parametrize("scenario_kwargs", DCN_SCENARIOS)
# def test_deep_cross_network(scenario_kwargs):

#     scenario_kwargs = scenario_kwargs.copy()
#     input_shape = scenario_kwargs.pop("input_shape")
#     output_size = scenario_kwargs.pop("output_size")
#     dummy_input = np.random.randn(32, *input_shape)
#     # check that correct errors are raised for some scenarios
#     if isinstance(scenario_kwargs["out_bias_init"], np.ndarray):
#         if scenario_kwargs["out_bias_init"].shape[-1] != output_size:
#             with pytest.raises(AssertionError):
#                 models.deep_cross_network(input_shape, output_size, **scenario_kwargs)
#             return
#         else:
#             model = models.deep_cross_network(
#                 input_shape, output_size, **scenario_kwargs
#             )

#     else:
#         model = models.deep_cross_network(input_shape, output_size, **scenario_kwargs)

#     _test_model(model)
#     _test_prediction(model, scenario_kwargs, dummy_input, output_size)
#     _test_prediction_prob(model, scenario_kwargs, dummy_input, output_size)
