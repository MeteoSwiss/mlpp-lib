from inspect import getmembers, isclass

import pytest
import tensorflow as tf
from keras.engine.functional import Functional

from mlpp_lib import models
from mlpp_lib import probabilistic_layers
from mlpp_lib.batching import get_tensor_dataset


LAYERS = [obj[0] for obj in getmembers(probabilistic_layers, isclass)]


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
    assert out.shape[2] == targets.sizes["variable"]
