import logging
from typing import Optional, Union, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
)
from tensorflow.keras import Model, initializers

from mlpp_lib.physical_layers import *
from mlpp_lib.probabilistic_layers import *

try:
    import tcn  # type: ignore
except ImportError:
    TCN_IMPORTED = False
else:
    TCN_IMPORTED = True


_LOGGER = logging.getLogger(__name__)


@tf.keras.saving.register_keras_serializable()
class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def _build_fcn_block(
    inputs,
    hidden_layers,
    batchnorm,
    activations,
    dropout,
    mc_dropout,
    skip_connection,
    idx=0,
):
    if mc_dropout and dropout is None:
        _LOGGER.warning("dropout=None, hence I will ignore mc_dropout=True")

    x = inputs
    for i, units in enumerate(hidden_layers):
        x = Dense(units, name=f"dense_{idx}_{i}")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activations[i])(x)
        if i < len(dropout) and 0.0 < dropout[i] < 1.0:
            if mc_dropout:
                x = MonteCarloDropout(dropout[i], name=f"mc_dropout_{idx}_{i}")(x)
            else:
                x = Dropout(dropout[i], name=f"dropout_{idx}_{i}")(x)

    if skip_connection:
        x = Dense(inputs.shape[1], name=f"skip_dense_{idx}")(x)
        x = Add(name=f"skip_add_{idx}")([x, inputs])
        x = Activation(activation=activations[-1], name=f"skip_activation_{idx}")(x)
    return x


def _build_fcn_output(x, output_size, probabilistic_layer, out_bias_init):
    # probabilistic prediction
    if probabilistic_layer:
        probabilistic_layer = globals()[probabilistic_layer]
        n_params = probabilistic_layer.params_size(output_size)
        if isinstance(out_bias_init, np.ndarray):
            out_bias_init = np.hstack(
                [out_bias_init, [0.0] * (n_params - out_bias_init.shape[0])]
            )
            out_bias_init = initializers.Constant(out_bias_init)

        x = Dense(n_params, bias_initializer=out_bias_init, name="dist_params")(x)
        outputs = probabilistic_layer(output_size, name="output")(x)

    # deterministic prediction
    else:
        if isinstance(out_bias_init, np.ndarray):
            out_bias_init = initializers.Constant(out_bias_init)

        outputs = Dense(output_size, bias_initializer=out_bias_init, name="output")(x)

    return outputs


def fully_connected_network(
    input_shape: tuple[int],
    output_size: int,
    hidden_layers: list,
    batchnorm: bool = False,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[str] = None,
    skip_connection: bool = False,
) -> Model:
    """
    Build a Fully Connected Neural Network.

    Parameters
    ----------
    input_shape: tuple[int]
        Shape of the input samples (not including batch size)
    output_size: int
        Number of target predictants.
    hidden_layers: list[int]
        List that is used to define the fully connected block. Each element creates
        a Dense layer with the corresponding units.
    batchnorm: bool
        Use batch normalization. Default is False.
    activations: str or list[str]
        (Optional) Activation function(s) for the Dense layer(s). See https://keras.io/api/layers/activations/#relu-function.
        If a string is passed, the same activation is used for all layers. Default is `relu`.
    dropout: float or list[float]
        (Optional) Dropout rate for the optional dropout layers. If a `float` is passed,
        dropout layers with the given rate are created after each Dense layer, except before the output layer.
        Default is None.
    mc_dropout: bool
        Enable Monte Carlo dropout during inference. It has no effect during training.
        It has no effect if `dropout=None`. Default is false.
    out_bias_init: str or np.ndarray
        (Optional) Specifies the initialization of the output layer bias. If a string is passed,
        it must be a valid Keras built-in initializer (see https://keras.io/api/layers/initializers/).
        If an array is passed, it must match the `output_size` argument.
    probabilistic_layer: str
        (Optional) Name of a probabilistic layer defined in `mlpp_lib.probabilistic_layers`, which is
        used as output layer of the keras `Model`. Default is None.
    skip_connection: bool
        Include a skip connection to the MLP architecture. Default is False.

    Return
    ------
    model: keras Functional model
        The built (but not yet compiled) model.
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

    if isinstance(out_bias_init, np.ndarray):
        out_bias_init_shape = out_bias_init.shape[-1]
        assert out_bias_init.shape[-1] == output_size, (
            f"Bias initialization array is shape {out_bias_init_shape}"
            f"but output size is {output_size}"
        )

    inputs = tf.keras.Input(shape=input_shape)
    x = _build_fcn_block(
        inputs,
        hidden_layers,
        batchnorm,
        activations,
        dropout,
        mc_dropout,
        skip_connection,
    )
    outputs = _build_fcn_output(x, output_size, probabilistic_layer, out_bias_init)
    model = Model(inputs=inputs, outputs=outputs)

    return model


def fully_connected_multibranch_network(
    input_shape: tuple[int],
    output_size: int,
    hidden_layers: list,
    batchnorm: bool = False,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[str] = None,
    skip_connection: bool = False,
) -> Model:
    """
    Build a multi-branch Fully Connected Neural Network.

    Parameters
    ----------
    input_shape: tuple[int]
        Shape of the input samples (not including batch size)
    output_size: int
        Number of target predictants.
    hidden_layers: list[int]
        List that is used to define the fully connected block. Each element creates
        a Dense layer with the corresponding units.
    batchnorm: bool
        Use batch normalization. Default is False.
    activations: str or list[str]
        (Optional) Activation function(s) for the Dense layer(s). See https://keras.io/api/layers/activations/#relu-function.
        If a string is passed, the same activation is used for all layers. Default is `relu`.
    dropout: float or list[float]
        (Optional) Dropout rate for the optional dropout layers. If a `float` is passed,
        dropout layers with the given rate are created after each Dense layer, except before the output layer.
        Default is None.
        mc_dropout: bool
        Enable Monte Carlo dropout during inference. It has no effect during training.
        It has no effect if `dropout=None`. Default is false.
    out_bias_init: str or np.ndarray
        (Optional) Specifies the initialization of the output layer bias. If a string is passed,
        it must be a valid Keras built-in initializer (see https://keras.io/api/layers/initializers/).
        If an array is passed, it must match the `output_size` argument.
    probabilistic_layer: str
        (Optional) Name of a probabilistic layer defined in `mlpp_lib.probabilistic_layers`, which is
        used as output layer of the keras `Model`. Default is None.
    skip_connection: bool
        Include a skip connection to the MLP architecture. Default is False.

    Return
    ------
    model: keras Functional model
        The built (but not yet compiled) model.
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

    if isinstance(out_bias_init, np.ndarray):
        out_bias_init_shape = out_bias_init.shape[-1]
        assert out_bias_init.shape[-1] == output_size, (
            f"Bias initialization array is shape {out_bias_init_shape}"
            f"but output size is {output_size}"
        )

    if probabilistic_layer:
        n_params = globals()[probabilistic_layer].params_size(output_size)
        n_branches = n_params
    else:
        n_branches = output_size

    inputs = tf.keras.Input(shape=input_shape)
    all_branch_outputs = []

    for idx in range(n_branches):
        x = _build_fcn_block(
            inputs,
            hidden_layers,
            batchnorm,
            activations,
            dropout,
            mc_dropout,
            skip_connection,
            idx,
        )
        all_branch_outputs.append(x)

    concatenated_x = tf.keras.layers.Concatenate()(all_branch_outputs)
    outputs = _build_fcn_output(
        concatenated_x, output_size, probabilistic_layer, out_bias_init
    )
    model = Model(inputs=inputs, outputs=outputs)

    return model


def deep_cross_network(
    input_shape: tuple[int],
    output_size: int,
    hidden_layers: list,
    batchnorm: bool = True,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[str] = None,
    skip_connection: bool = False,
):
    """
    Build a Deep and Cross Network (see https://arxiv.org/abs/1708.05123).

    Parameters
    ----------
    input_shape: tuple[int]
        Shape of the input samples (not including batch size)
    output_size: int
        Number of target predictants.
    hidden_layers: list[int]
        List that is used to define the fully connected block. Each element creates
        a Dense layer with the corresponding units.
    batchnorm: bool
        Use batch normalization. Default is True.
    activations: str or list[str]
        (Optional) Activation function(s) for the Dense layer(s). See https://keras.io/api/layers/activations/#relu-function.
        If a string is passed, the same activation is used for all layers. Default is `relu`.
    dropout: float or list[float]
        (Optional) Dropout rate for the optional dropout layers. If a `float` is passed,
        dropout layers with the given rate are created after each Dense layer, except before the output layer.
        Default is None.
    mc_dropout: bool
        Enable Monte Carlo dropout during inference. It has no effect during training.
        It has no effect if `dropout=None`. Default is false.
    out_bias_init: str or np.ndarray
        (Optional) Specifies the initialization of the output layer bias. If a string is passed,
        it must be a valid Keras built-in initializer (see https://keras.io/api/layers/initializers/).
        If an array is passed, it must match the `output_size` argument.
    probabilistic_layer: str
        (Optional) Name of a probabilistic layer defined in `mlpp_lib.probabilistic_layers`, which is
        used as output layer of the keras `Model`. Default is None.
    skip_connection: bool
        Include a skip connection to the MLP architecture. Default is False.

    Return
    ------
    model: keras Functional model
        The built (but not yet compiled) model.
    """
    if isinstance(dropout, list):
        assert len(dropout) == len(hidden_layers)
    elif isinstance(dropout, float):
        dropout = [dropout] * (len(hidden_layers))
    else:
        dropout = []

    if isinstance(activations, list):
        assert len(activations) == len(hidden_layers)
    elif isinstance(activations, str):
        activations = [activations] * len(hidden_layers)

    if isinstance(out_bias_init, np.ndarray):
        out_bias_init_shape = out_bias_init.shape[-1]
        assert out_bias_init.shape[-1] == output_size, (
            f"Bias initialization array is shape {out_bias_init_shape}"
            f"but output size is {output_size}"
        )

    # cross part
    inputs = tf.keras.layers.Input(shape=input_shape)
    cross = inputs
    for _ in hidden_layers:
        units_ = cross.shape[-1]
        x = Dense(units_)(cross)
        cross = inputs * x + cross
    cross = BatchNormalization()(cross)
    # cross = tf.keras.Model(inputs=inputs, outputs=cross, name="crossblock")

    # deep part
    deep = inputs
    deep = _build_fcn_block(
        deep,
        hidden_layers,
        batchnorm,
        activations,
        dropout,
        mc_dropout,
        skip_connection=False,
    )

    # merge
    merge = tf.keras.layers.Concatenate()([cross, deep])

    if skip_connection:
        merge = Dense(input_shape[0])(merge)
        merge = Add()([merge, inputs])
        merge = Activation(activation=activations[-1])(merge)

    # probabilistic prediction
    if probabilistic_layer:
        probabilistic_layer = globals()[probabilistic_layer]
        n_params = probabilistic_layer.params_size(output_size)
        if isinstance(out_bias_init, np.ndarray):
            out_bias_init = np.hstack(
                [out_bias_init, [0.0] * (n_params - out_bias_init.shape[0])]
            )
            out_bias_init = initializers.Constant(out_bias_init)

        x = Dense(n_params, bias_initializer=out_bias_init, name="dist_params")(merge)
        outputs = probabilistic_layer(output_size, name="output")(x)

    # deterministic prediction
    else:
        if isinstance(out_bias_init, np.ndarray):
            out_bias_init = initializers.Constant(out_bias_init)

        outputs = Dense(output_size, bias_initializer=out_bias_init, name="output")(
            merge
        )

    model = Model(inputs=inputs, outputs=outputs, name="deep_cross_network")
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
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    **kwargs,
) -> Model:
    """
    Build a Temporal Convolutional Network.
    """

    if not TCN_IMPORTED:
        raise ImportError("Optional dependency keras-tcn is missing!")

    if isinstance(out_bias_init, np.ndarray):
        out_bias_init = initializers.Constant(out_bias_init)

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

    outputs = Dense(output_size, bias_initializer=out_bias_init, name="output")(x_tcn)
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
