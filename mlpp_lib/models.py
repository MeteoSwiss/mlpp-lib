import tensorflow as tf
from typing import Optional, Union, Any
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras import Model, initializers

from mlpp_lib.physical_layers import *
import numpy as np

try:
    import tcn  # type: ignore
except ImportError:
    TCN_IMPORTED = False
else:
    TCN_IMPORTED = True


def fully_connected_network(
    input_shape: tuple[int],
    output_size: int,
    hidden_layers: list = [32, 32, 32],
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    output_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
) -> Model:
    """
    Build a Fully Connected Neural Network.
    """

    if isinstance(dropout, list):
        assert len(dropout) == len(hidden_layers)
    elif isinstance(dropout, float):
        dropout = [dropout] * (len(hidden_layers) - 1)
    else:
        dropout = []

    if isinstance(activations, list):
        assert len(activations) == len(hidden_layers)
    elif isinstance(activations, str):
        activations = [activations] * len(hidden_layers)

    if isinstance(output_bias_init, np.ndarray):
        output_bias_init = initializers.Constant(output_bias_init)

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = Dense(units, activation=activations[i], name=f"dense_{i}")(x)
        if i < len(dropout):
            x = Dropout(dropout[i], name=f"dropout_{i}")(x)

    outputs = Dense(output_size, bias_initializer=output_bias_init, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def temporal_convolutional_network(
    input_shape: tuple[int],
    output_size: int,
    nb_filters: int,
    kernel_size: int = 3,
    dilations: tuple[int] = (1, 2, 4),
    use_skip_connections: bool = True,
    dropout_rate: float = 0,
    activation: str = "relu",
    output_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    **kwargs,
) -> Model:
    """
    Build a Temporal Convolutional Network.
    """

    if not TCN_IMPORTED:
        raise ImportError("Optional dependency keras-tcn is missing!")

    if isinstance(output_bias_init, np.ndarray):
        output_bias_init = initializers.Constant(output_bias_init)

    inputs = tf.keras.Input(shape=input_shape, name="input")
    x_tcn = tcn.TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        use_skip_connections=use_skip_connections,
        activation=activation,
        dropout_rate=dropout_rate,
        padding="same",
        return_sequences=True,
        name="tcn",
        **kwargs,
    )(inputs)

    outputs = Dense(output_size, bias_initializer=output_bias_init, name="output")(
        x_tcn
    )
    model = Model(inputs=inputs, outputs=outputs)

    return model


def architecture_constrained_fcn(
    input_shape: tuple[int],
    direct_output_size: int,
    physical_layer: str = "ThermodynamicLayer",
    **kwargs,
) -> Model:
    """
    Build a Fully Connected Neural Network with a physical layer.
    """

    fully_connected_block = fully_connected_network(
        input_shape=input_shape, output_size=direct_output_size, **kwargs
    )

    inputs = fully_connected_block.input
    direct_outputs = fully_connected_block.output
    outputs = globals()[physical_layer]()(direct_outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def architecture_constrained_tcn(
    input_shape: tuple[int],
    direct_output_size: int,
    physical_layer: str = "ThermodynamicLayer",
    **kwargs,
) -> Model:
    """
    Build a Temporal Convolutional Network with a physical layer.
    """

    temporal_convolutional_block = temporal_convolutional_network(
        input_shape=input_shape, output_size=direct_output_size, **kwargs
    )

    inputs = temporal_convolutional_block.input
    direct_outputs = temporal_convolutional_block.output
    outputs = globals()[physical_layer]()(direct_outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model
