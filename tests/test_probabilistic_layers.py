from inspect import getmembers

import tensorflow as tf
from mlpp_lib import probabilistic_layers

import pytest

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
