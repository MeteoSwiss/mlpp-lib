import subprocess
from inspect import getmembers

import numpy as np
import pytest
import tensorflow as tf
from keras.engine.functional import Functional

from mlpp_lib import models
from mlpp_lib import probabilistic_layers
from mlpp_lib.batching import get_tensor_dataset


LAYERS = [
    obj[0] for obj in getmembers(probabilistic_layers) if hasattr(obj[1], "_is_layer")
]


@pytest.mark.parametrize("layer", LAYERS)
def test_probabilistic_layers(layer):

    layer_class = getattr(probabilistic_layers, layer)

    output_size = 2
    params_size = layer_class.params_size(output_size)

    # build
    layer = layer_class(output_size)

    # call
    input_tensor = tf.random.normal((10, params_size))
    output_dist = layer(input_tensor)

    n_samples = 5
    samples = output_dist.sample(n_samples)

    assert samples.shape == (n_samples, 10, output_size)


@pytest.mark.parametrize("layer", LAYERS)
def test_probabilistic_model(layer):
    """Test model compile with prob layers"""

    layer_class = getattr(probabilistic_layers, layer)
    tfkl = tf.keras.layers
    input_shape = [28, 28, 1]
    encoded_shape = 2
    encoder = tf.keras.Sequential(
        [
            tfkl.InputLayer(input_shape=input_shape),
            tfkl.Flatten(),
            tfkl.Dense(10, activation="relu"),
            tfkl.Dense(layer_class.params_size(encoded_shape)),
            layer_class(encoded_shape),
        ]
    )
    encoder.summary()
    encoder.compile()
    assert isinstance(encoder, Functional)


@pytest.mark.parametrize("layer", LAYERS)
def test_probabilistic_model_predict(layer, features_dataset, targets_dataset):
    features, targets = get_tensor_dataset(features_dataset, targets_dataset, [])
    x_shape = features.shape
    y_shape = targets.shape
    input_shape = x_shape[1]
    output_size = y_shape[1]
    model = models.fully_connected_network(
        input_shape, output_size, hidden_layers=[3], probabilistic_layer=layer
    )
    num_samples = 10
    out = model(features.values).sample(num_samples).numpy()
    assert out.shape[0] == num_samples
    assert out.shape[1] == features.sizes["sample"]
    assert out.shape[2] == targets.sizes["target"]


@pytest.mark.parametrize("save_format", ["tf", "h5"])
@pytest.mark.parametrize("layer", LAYERS)
def test_probabilistic_save_model(save_format, layer, tmp_path):
    """Test model save/load with prob layers"""

    if save_format == "h5":
        tmp_path = f"{tmp_path}.h5"
        save_traces = True  # default value
    else:
        tmp_path = f"{tmp_path}"
        save_traces = False

    model = models.fully_connected_network(
        (5,), 2, hidden_layers=[3], probabilistic_layer=layer
    )
    assert isinstance(model.from_config(model.get_config()), Functional)
    model.compile()
    model.save(tmp_path, save_traces=save_traces)

    # test trying to load the model from a new process
    # this is a bit slow, since each process needs to reload all the dependencies ...
    args = [
        "python",
        "-c",
        f"import tensorflow as tf; from mlpp_lib.probabilistic_layers import {layer}; tf.keras.models.load_model('{tmp_path}')",
    ]
    completed_process = subprocess.run(args)
    assert completed_process.returncode == 0, "failed to reload model"

    # loading here is not a good test because the custom layers are still somehow in memory
    # we'll do it anyway to test that the behavior doesn't change after loading
    input_arr = tf.random.uniform((1, 5))
    outputs = model(input_arr).mean()
    del model
    model = tf.keras.models.load_model(tmp_path)
    assert isinstance(model, Functional)
    np.testing.assert_allclose(model(input_arr).mean(), outputs)
