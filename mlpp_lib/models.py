import logging
from typing import Optional, Union, Any, Literal

import numpy as np
# import tensorflow as tf
import keras
from keras.src.layers import (
    Add,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Concatenate
)
from keras import Model, initializers

from mlpp_lib.physical_layers import *
# from mlpp_lib import probabilistic_layers
from mlpp_lib.probabilistic_layers import BaseDistributionLayer, BaseParametricDistributionModule, distribution_to_layer
from mlpp_lib.layers import FullyConnectedLayer, MultibranchLayer, CrossNetLayer, ParallelConcatenateLayer

try:
    import tcn  # type: ignore
except ImportError:
    TCN_IMPORTED = False
else:
    TCN_IMPORTED = True


_LOGGER = logging.getLogger(__name__)

class ProbabilisticModel(keras.Model):
    """ A probabilistic model composed of an encoder layer 
    and a probabilistic layer predicting the output's distribution.
    """
    def __init__(self, encoder_layer: keras.Layer, probabilistic_layer: BaseDistributionLayer, default_output_type: Literal["distribution", "samples"] = "distribution"):
        """_summary_

        Args:
            encoder_layer (keras.Layer): The encoder layer, transforming the inputs into 
            some latent dimension.
            probabilistic_layer (BaseDistributionLayer): the output layer predicting the distribution.
            default_output_type (Literal[distribution, samples], optional): Defines the defult behaviour of self.call(), where the model can either output a parametric 
            distribution, or samples obtained from it. This is important to when fitting the model, as the type of output defines what loss functions are suitable. 
            Defaults to "distribution".
        """
        super().__init__()
        
        self.encoder_layer = encoder_layer
        self.probabilistic_layer = probabilistic_layer
        self.default_output_type = default_output_type
        
        
    def call(self, inputs, output_type: Optional[Literal["distribution", "samples"]] = None):
        if output_type is None:
            output_type = self.default_output_type
            
        enc = self.encoder_layer(inputs)
        return self.probabilistic_layer(enc, output_type=output_type)
    



@keras.saving.register_keras_serializable()
class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

def get_probabilistic_layer(distribution: str, bias_init, distribution_kwargs={},num_samples=21):
    probabilistic_layer = distribution_to_layer[distribution](**distribution_kwargs)
    return BaseDistributionLayer(distribution=probabilistic_layer,
                            num_samples=num_samples,
                            bias_init=bias_init)

# def get_probabilistic_layer(
#     output_size,
#     probabilistic_layer: Union[str, dict]
# ) -> Callable:
#     """Get the probabilistic layer."""

#     if isinstance(probabilistic_layer, dict):
#         probabilistic_layer_name = list(probabilistic_layer.keys())[0]
#         probabilistic_layer_options = probabilistic_layer[probabilistic_layer_name]
#     else:
#         probabilistic_layer_name = probabilistic_layer
#         probabilistic_layer_options = {}

#     if hasattr(probabilistic_layers, probabilistic_layer_name):
#         _LOGGER.info(f"Using custom probabilistic layer: {probabilistic_layer_name}")
#         probabilistic_layer_obj = getattr(probabilistic_layers, probabilistic_layer_name)
#         n_params = getattr(probabilistic_layers, probabilistic_layer_name).params_size(output_size)
#         probabilistic_layer = (
#             probabilistic_layer_obj(output_size, name="output", **probabilistic_layer_options) if isinstance(probabilistic_layer_obj, type) 
#             else probabilistic_layer_obj(output_size, name="output")
#         )
#     else:
#         raise KeyError(f"The probabilistic layer {probabilistic_layer_name} is not available.")

#     return probabilistic_layer, n_params


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


def _build_fcn_output(output_size, out_bias_init, probabilistic_layer=None, **distribution_kwargs):
    if probabilistic_layer is None:
        if isinstance(out_bias_init, np.ndarray):
            out_bias_init = initializers.Constant(out_bias_init)
        return Dense(output_size, name='output', bias_initializer=out_bias_init)
    
    
    prob_layer = get_probabilistic_layer(distribution=probabilistic_layer, bias_init=out_bias_init,distribution_kwargs=distribution_kwargs)
    return prob_layer
 
def fully_connected_network(
    output_size: int,
    hidden_layers: list,
    batchnorm: bool = False,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[str] = None,
    skip_connection: bool = False,
    prob_layer_kwargs: dict = {}
) -> Model:
    """
    Get an unbuilt Fully Connected Neural Network.

    Parameters
    ----------
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
    model: keras model
        The built (but not yet compiled) model.
    """
    
    ffnn = FullyConnectedLayer(hidden_layers=hidden_layers,
                               batchnorm=batchnorm,
                               activations=activations,
                               dropout=dropout,
                               mc_dropout=mc_dropout,
                               skip_connection=skip_connection)
    
    output_layer = _build_fcn_output(out_bias_init=out_bias_init,
                                     output_size=output_size,
                                     probabilistic_layer=probabilistic_layer, **prob_layer_kwargs)
    
    if probabilistic_layer is None:
        return keras.models.Sequential([ffnn, output_layer])
    
    return ProbabilisticModel(encoder_layer=ffnn,
                               probabilistic_layer=output_layer)



def fully_connected_multibranch_network(
    output_size: int,
    hidden_layers: list,
    n_branches,
    batchnorm: bool = False,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[str] = None,
    skip_connection: bool = False,
    aggregation: Literal['sum', 'concat']='concat',
    prob_layer_kwargs: dict = {}
) -> Model:
    """
    Returns an unbuilt a multi-branch Fully Connected Neural Network.

    Parameters
    ----------
    input_shape: tuple[int]
        Shape of the input samples (not including batch size)
    output_size: int
        Number of target predictants.
    hidden_layers: list[int]
        List that is used to define the fully connected block. Each element creates
        a Dense layer with the corresponding units.
    n_branches: int
        The number of branches.
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
    aggregation: Literal['sum', 'concat']
        The aggregation strategy to combine the branches' outputs.

    Return
    ------
    model: keras model
        The unbuilt and uncompiled model.
    """
    
    branch_layers = []

    for idx in range(n_branches):
        branch_layers.append(FullyConnectedLayer(
            hidden_layers=hidden_layers,
            batchnorm=batchnorm,
            activations=activations,
            dropout=dropout,
            mc_dropout=mc_dropout,
            skip_connection=skip_connection,
            indx=idx
        ))
        
    mb_ffnn = MultibranchLayer(branches=branch_layers, aggregation=aggregation)
    
    output_layer = _build_fcn_output(out_bias_init=out_bias_init,
                                     output_size=output_size,
                                     probabilistic_layer=probabilistic_layer, **prob_layer_kwargs)
    
    if probabilistic_layer is None:
        return keras.models.Sequential([mb_ffnn, output_layer])
    
    return ProbabilisticModel(encoder_layer=mb_ffnn,
                               probabilistic_layer=output_layer)


def deep_cross_network(
    output_size: int,
    hidden_layers: list,
    n_cross_layers: int,
    cross_layers_hiddensize: int,
    batchnorm: bool = True,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[str] = None,
    prob_layer_kwargs: dict = {}
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
    n_cross_layers: int
        The number of cross layers
    cross_layers_hiddensize: int
        The hidden size to be used in the cross layers
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

    Return
    ------
    model: keras model
        The built (but not yet compiled) model.
    """


    # cross part
    cross_layer = CrossNetLayer(hidden_size=cross_layers_hiddensize,
                                depth=n_cross_layers)
    
    # deep part
    
    deep_layer = FullyConnectedLayer(hidden_layers=hidden_layers,
                                     batchnorm=batchnorm,
                                     activations=activations,
                                     dropout=dropout,
                                     mc_dropout=mc_dropout)

    
    encoder = ParallelConcatenateLayer([cross_layer, deep_layer])

    output_layer = _build_fcn_output(out_bias_init=out_bias_init,
                                     output_size=output_size,
                                     probabilistic_layer=probabilistic_layer, **prob_layer_kwargs)

    if probabilistic_layer is None:
        return keras.models.Sequential([encoder, output_layer])
    
    return ProbabilisticModel(encoder_layer=encoder,
                               probabilistic_layer=output_layer)


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

    inputs = keras.Input(shape=input_shape, name="input")
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
