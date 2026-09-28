from inspect import getmembers, isclass, getmodule
import pytest
from keras.layers import Layer
import keras
import numpy as np

from mlpp_lib import layers

ALL_LAYERS = {
    obj[0]: obj[1]
    for obj in getmembers(layers, isclass)
    if issubclass(obj[1], Layer) and getmodule(obj[1]) == layers
}


layers_args = {
    "MultilayerPerceptron": {
        "hidden_layers": [16, 8],
    },
    "MonteCarloDropout": {"rate": 0.5},
    "MultibranchLayer": {
        "branches": [
            layers.MultilayerPerceptron(hidden_layers=[16, 8]),
            layers.MultilayerPerceptron(hidden_layers=[16, 4]),
        ]
    },
    "CrossNetLayer": {"hidden_size": 16, "depth": 2},
    "CrossNetLayerNoProjection": {"depth": 2},
    "ParallelConcatenateLayer": {
        "layers": [
            layers.MultilayerPerceptron(hidden_layers=[16, 8]),
            layers.MultilayerPerceptron(hidden_layers=[16, 4]),
        ]
    },
}

ALL_LAYERS["CrossNetLayerNoProjection"] = layers.CrossNetLayer


@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_single_layer_serialization(layer, tmp_path):
    """
    Build a model consisting of a single layer for all layers and
    test serialization and deserialization
    """
    data = keras.random.normal((32, 4))

    layer_cls = ALL_LAYERS[layer]
    # get layer
    layer = layer_cls(**layers_args[layer])

    # make a model that can be serialized
    inputs = keras.Input(shape=(4,))
    outputs = layer(inputs)
    model = keras.Model(inputs, outputs)

    model.compile()

    tmp_path = f"{tmp_path}.keras"
    model.save(tmp_path)
    # print(model.summary())
    loaded_model = keras.saving.load_model(tmp_path)
    loaded_model.compile()

    # all weight matrices must be the same
    for w1, w2 in zip(model.trainable_weights, loaded_model.trainable_weights):
        assert (w1 == w2).all()

    keras.utils.set_random_seed(42)  # for dropouts
    model_out = model(data)
    keras.utils.set_random_seed(42)  # for dropouts
    loaded_model_out = loaded_model(data)

    # check again that after feeding in new data, the weights are still the same.
    # This ensures that no layer in "unbuilt" and only gets built as the first data is given.
    for w1, w2 in zip(model.trainable_weights, loaded_model.trainable_weights):
        assert (w1 == w2).all()

    # check that the two outputs are the same
    assert keras.ops.isclose(model_out, loaded_model_out).all()


@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_gradient_flow(layer):

    def get_trainable_weights(layer):

        weights = []

        if isinstance(layer, layers.Layer):

            if layer.trainable_weights:
                weights.extend(layer.trainable_weights)

            for sub_layer in layer._layers:
                weights.extend(get_trainable_weights(sub_layer))

        return weights

    layer_cls = ALL_LAYERS[layer]
    layer = layer_cls(**layers_args[layer])

    inputs = keras.Input(shape=(4,))
    outputs = layer(inputs)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="mse")

    x = keras.random.normal((32, 4))
    y = keras.random.normal((32, model.output_shape[1]))

    # Check that before training, no parameter has a gradient tensor
    trainable_params = get_trainable_weights(layer)
    if keras.backend.backend() == "torch":
        for param in trainable_params:
            assert param.value.grad is None
    else:
        pytest.fail(f"Implement gradient checking for {keras.backend.backend()}")

    model.fit(x, y, epochs=1, batch_size=5)

    # Check that after training, every parameter has a gradient tensor
    trainable_params = get_trainable_weights(layer)
    if keras.backend.backend() == "torch":
        for param in trainable_params:
            assert param.value.grad is not None
    else:
        pytest.fail(f"Implement gradient checking for {keras.backend.backend()}")


@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_batch_size_one(layer):
    layer = ALL_LAYERS[layer](**layers_args[layer])
    out = layer(keras.random.normal((1, 4)))
    assert out.shape[0] == 1
    assert tuple(out.shape) == tuple(layer.compute_output_shape((1, 4)))


def test_cross_net_layer():
    x = keras.random.normal((8, 4))
    layer = layers.CrossNetLayer(depth=1)
    out = layer(x)
    w, b = layer.ws[0], layer.bs[0]
    # x_1 = x_0 * (x_0^T w) + b + x_0
    expected = x * keras.ops.matmul(x, w) + b + x
    assert keras.ops.all(keras.ops.isclose(out, expected))
