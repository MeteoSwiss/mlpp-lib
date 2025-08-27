import subprocess
from inspect import getmembers, isfunction, isclass

import numpy as np
import pytest
from keras import Model
import torch
import keras
from mlpp_lib import models
from mlpp_lib import losses, metrics
from mlpp_lib import probabilistic_layers
from mlpp_lib.utils import get_loss, get_metric, get_optimizer


def _belongs_here(obj, module):
    return obj[1].__module__ == module.__name__

from mlpp_lib.probabilistic_layers import distribution_to_layer

ALL_PROB_LAYERS = list(distribution_to_layer.keys())

ALL_LOSSES = [
    obj[0]
    for obj in getmembers(losses, isfunction) + getmembers(losses, isclass)
    if _belongs_here(obj, losses)
]

ALL_METRICS = [
    obj[0]
    for obj in getmembers(metrics, isfunction) + getmembers(metrics, isclass)
    if _belongs_here(obj, metrics)
]


TEST_LOSSES = [
    {"CRPSEnsemble": {"num_samples": 10}},
]

TEST_METRICS = [
    "expected_bias",
    "expected_mean_absolute_error",
    {"MAEBusts": {"threshold": 0.5}},
]

@pytest.mark.parametrize("loss", TEST_LOSSES)
@pytest.mark.parametrize("prob_layer", ALL_PROB_LAYERS)
def test_save_model(loss, prob_layer, tmp_path):
    """Test model save/load"""

    
    tmp_path = f"{tmp_path}.keras"

    prob_layer_kwargs = {}
    if prob_layer == 'multivariate_tril_gaussian':
        prob_layer_kwargs['dim'] = 3
    elif prob_layer == 'censored_gaussian':
        prob_layer_kwargs['a'] = 0.0
        prob_layer_kwargs['b'] = 10.0
    elif prob_layer == 'truncated_gaussian':
        prob_layer_kwargs['a'] = 0.0
        prob_layer_kwargs['b'] = 10.0
        
    model = models.fully_connected_network(
        2,
        hidden_layers=[3],
        probabilistic_layer=prob_layer,
        mc_dropout=False,
        prob_layer_kwargs=prob_layer_kwargs
    )
    
    input_data = keras.random.normal((32,5))
    output_data = model(input_data).mean
    
    # The assertion below fails because of safety mechanism in keras against
    # the deserialization of Lambda layers that we cannot switch off
    # assert isinstance(model.from_config(model.get_config()), Model)
    loss = get_loss(loss)
    metrics = [get_metric(metric) for metric in TEST_METRICS]
    model.compile(loss=loss, metrics=metrics)

    model.save(tmp_path)
    
    loaded_model = keras.saving.load_model(tmp_path)
    
    # Check all weights are the same
    w1 = model.trainable_weights
    w2 = loaded_model.trainable_weights

    assert len(w1) == len(w2)

    for a, b in zip(w1, w2):
        assert a.shape == b.shape
        assert np.allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-8)
    
    # check output mean matches between the two models
    assert torch.allclose(model(input_data).mean, loaded_model(input_data).mean)
    
    # test trying to load the model from a new process
    # this is a bit slow, since each process needs to reload all the dependencies ...

    # # not compiling
    # args = [
    #     "python",
    #     "-c",
    #     "import tensorflow as tf;"
    #     f"from mlpp_lib.probabilistic_layers import {prob_layer};"
    #     f"tf.keras.saving.load_model('{tmp_path}', compile=False, safe_mode=False)",
    # ]
    # completed_process = subprocess.run(args, shell=True)
    # assert completed_process.returncode == 0, "failed to reload model"

    # # compiling
    # args = [
    #     "python",
    #     "-c",
    #     "import tensorflow as tf;"
    #     f"from mlpp_lib.losses import {loss};"
    #     f"from mlpp_lib.probabilistic_layers import {prob_layer};"
    #     f"tf.keras.saving.load_model('{tmp_path}', custom_objects={{'{loss}':{loss}}}, safe_mode=False)",
    # ]
    # completed_process = subprocess.run(args, shell=True)
    # assert completed_process.returncode == 0, "failed to reload model"

    # input_arr = tf.random.uniform((1, 5))
    # pred1 = model(input_arr)
    # del model
    # tf.keras.backend.clear_session()
    # model = tf.keras.saving.load_model(tmp_path, compile=False, safe_mode=False)
    # assert isinstance(model, Model)

    # pred2 = model(input_arr)
    # try:
    #     # Idependent layers have a "distribution" attribute
    #     pred1_params = pred1.parameters["distribution"].parameters
    #     pred2_params = pred2.parameters["distribution"].parameters
    # except KeyError:
    #     pred1_params = pred1.parameters
    #     pred2_params = pred2.parameters

    # for param in pred1_params.keys():
    #     try:
    #         param_array1 = pred1_params[param].numpy()
    #         param_array2 = pred2_params[param].numpy()
    #     except AttributeError:
    #         continue

    #     np.testing.assert_allclose(param_array1, param_array2)


# def test_save_model_mlflow(tmp_path):
#     """Test model save/load"""
#     pytest.importorskip("mlflow")

#     import mlflow

#     mlflow_uri = f"file://{tmp_path.absolute()}/mlruns"
#     mlflow.set_tracking_uri(mlflow_uri)

#     model = models.fully_connected_network(
#         (5,),
#         2,
#         hidden_layers=[3, 3],
#         dropout=0.5,
#         mc_dropout=True,
#         probabilistic_layer="IndependentNormal",
#     )
#     optimizer = get_optimizer("Adam")
#     model.compile(optimizer=optimizer, loss=None, metrics=None)
#     custom_objects = tf.keras.layers.serialize(model)

#     model_info = mlflow.tensorflow.log_model(
#         model,
#         "model_save",
#         custom_objects=custom_objects,
#         keras_model_kwargs={"save_format": "keras"},
#     )

#     tf.keras.backend.clear_session()

#     # this raises a ValueError because of the risk of deserializing Lambda layers
#     with pytest.raises(ValueError):
#         model = mlflow.tensorflow.load_model(model_info.model_uri)

#     # this should work
#     model: tf.tensorflow.Model = mlflow.tensorflow.load_model(
#         model_info.model_uri, keras_model_kwargs={"safe_mode": False}
#     )

#     assert isinstance(model, Model)
