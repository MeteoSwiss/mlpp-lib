import logging
import warnings
from typing import Any, Literal, Optional, Union

import keras
import numpy as np
from keras import Model, initializers
from keras.layers import Dense

from mlpp_lib.layers import (
    CrossNetLayer,
    MonteCarloDropout,  # noqa: F401 (kept importable from here)
    MultibranchLayer,
    MultilayerPerceptron,
    ParallelConcatenateLayer,
)
from mlpp_lib.probabilistic_layers import (
    DistributionLayer,
    get_distribution,
)

_LOGGER = logging.getLogger(__name__)

OutputType = Literal["distribution", "samples", "expected"]


@keras.saving.register_keras_serializable(package="mlpp_lib")
class ProbabilisticModel(keras.Model):
    """A probabilistic model composed of an encoder layer
    and a probabilistic layer predicting the output's distribution.

    Parameters
    ----------
    encoder: keras.Layer
        The encoder layer, transforming the inputs into some latent dimension.
    output_distribution: DistributionLayer
        The output layer predicting the distribution.
    default_output_type: "distribution", "samples" or "expected"
        Defines the default behaviour of `call()`, where the model can either output
        a parametric distribution, samples obtained from it, or the expected value.
        This is important when fitting the model, as the type of output defines what
        loss functions are suitable. Defaults to "distribution".
    """

    def __init__(
        self,
        encoder: keras.Layer,
        output_distribution: DistributionLayer,
        default_output_type: OutputType = "distribution",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.encoder = encoder
        self.output_distribution = output_distribution
        self.default_output_type = default_output_type

    def call(
        self,
        inputs,
        output_type: Optional[OutputType] = None,
        num_samples: Optional[int] = None,
        training=None,
    ):
        if output_type is None:
            output_type = self.default_output_type

        enc = self.encoder(inputs, training=training)
        return self.output_distribution(
            enc, output_type=output_type, num_samples=num_samples, training=training
        )

    def compute_output_shape(self, input_shape):
        return self.output_distribution.compute_output_shape(
            self.encoder.compute_output_shape(input_shape)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder": keras.layers.serialize(self.encoder),
                "output_distribution": keras.layers.serialize(self.output_distribution),
                "default_output_type": self.default_output_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Creates an instance of the model from its config."""
        encoder = keras.layers.deserialize(config.pop("encoder"))
        output_distribution = keras.layers.deserialize(
            config.pop("output_distribution")
        )
        return cls(encoder=encoder, output_distribution=output_distribution, **config)


def _parse_probabilistic_layer(
    probabilistic_layer: Union[str, dict], distribution_kwargs: Optional[dict] = None
) -> tuple[str, dict]:
    """Supports both `"Normal"` and the `{"Normal": {options}}` notation."""
    distribution_kwargs = dict(distribution_kwargs or {})
    if isinstance(probabilistic_layer, dict):
        if len(probabilistic_layer) != 1:
            raise ValueError(
                "probabilistic_layer must contain exactly one distribution name."
            )
        (name, options), *_ = probabilistic_layer.items()
        distribution_kwargs = {**(options or {}), **distribution_kwargs}
    else:
        name = probabilistic_layer
    return name, distribution_kwargs


def get_probabilistic_layer(
    distribution: Union[str, dict],
    output_size: int = 1,
    bias_init="zeros",
    distribution_kwargs: Optional[dict] = None,
    num_samples: int = 21,
) -> DistributionLayer:
    """Get the probabilistic output layer for the distribution with the given name
    (see `mlpp_lib.probabilistic_layers.DISTRIBUTIONS`)."""
    name, distribution_kwargs = _parse_probabilistic_layer(
        distribution, distribution_kwargs
    )
    return DistributionLayer(
        distribution=get_distribution(
            name, event_size=output_size, **distribution_kwargs
        ),
        num_samples=num_samples,
        bias_init=bias_init,
        name="output",
    )


def _num_distribution_parameters(
    probabilistic_layer: Union[str, dict], output_size: int, distribution_kwargs
) -> int:
    name, distribution_kwargs = _parse_probabilistic_layer(
        probabilistic_layer, distribution_kwargs
    )
    with warnings.catch_warnings():
        # deprecation warnings are raised when building the output layer
        warnings.simplefilter("ignore", DeprecationWarning)
        distribution = get_distribution(
            name, event_size=output_size, **distribution_kwargs
        )
    return distribution.num_parameters


def _expand_dropout(dropout, num_layers: int):
    """A float dropout rate is applied after each hidden layer except the last."""
    if isinstance(dropout, float):
        return [dropout] * (num_layers - 1)
    return dropout


def _build_output_layer(
    output_size: int,
    out_bias_init,
    probabilistic_layer: Optional[Union[str, dict]] = None,
    prob_layer_kwargs: Optional[dict] = None,
    num_samples: int = 21,
):
    if isinstance(out_bias_init, np.ndarray) and out_bias_init.shape[-1] != output_size:
        raise ValueError(
            f"Bias initialization array is shape {out_bias_init.shape} "
            f"but output size is {output_size}"
        )
    if probabilistic_layer is None:
        if isinstance(out_bias_init, np.ndarray):
            out_bias_init = initializers.Constant(out_bias_init)
        return Dense(output_size, name="output", bias_initializer=out_bias_init)

    return get_probabilistic_layer(
        distribution=probabilistic_layer,
        output_size=output_size,
        bias_init=out_bias_init,
        distribution_kwargs=prob_layer_kwargs,
        num_samples=num_samples,
    )


def _assemble(encoder: keras.Layer, output_layer: keras.Layer, name: str) -> Model:
    if isinstance(output_layer, DistributionLayer):
        return ProbabilisticModel(
            encoder=encoder, output_distribution=output_layer, name=name
        )
    return keras.models.Sequential([encoder, output_layer], name=name)


def fully_connected_network(
    output_size: int,
    hidden_layers: list,
    batchnorm: bool = False,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[Union[str, dict]] = None,
    skip_connection: bool = False,
    prob_layer_kwargs: Optional[dict] = None,
    num_samples: int = 21,
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
    probabilistic_layer: str or dict
        (Optional) Name of a distribution defined in `mlpp_lib.probabilistic_layers.DISTRIBUTIONS`,
        which is used as output layer of the keras `Model`, or a dictionary `{name: options}`.
        Default is None.
    skip_connection: bool
        Include a skip connection to the MLP architecture. Default is False.
    prob_layer_kwargs: dict
        (Optional) Options of the distribution, e.g. `{"low": 0.0}` for a `TruncatedNormal`.
    num_samples: int
        Default number of samples drawn when the model outputs samples.

    Return
    ------
    model: keras model
        The unbuilt (and not yet compiled) model.
    """

    ffnn = MultilayerPerceptron(
        hidden_layers=hidden_layers,
        batchnorm=batchnorm,
        activations=activations,
        dropout=_expand_dropout(dropout, len(hidden_layers)),
        mc_dropout=mc_dropout,
        skip_connection=skip_connection,
    )

    output_layer = _build_output_layer(
        output_size,
        out_bias_init,
        probabilistic_layer,
        prob_layer_kwargs,
        num_samples,
    )

    return _assemble(ffnn, output_layer, name="fully_connected_network")


def fully_connected_multibranch_network(
    output_size: int,
    hidden_layers: list,
    n_branches: Optional[int] = None,
    batchnorm: bool = False,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[Union[str, dict]] = None,
    skip_connection: bool = False,
    aggregation: Literal["sum", "concat"] = "concat",
    prob_layer_kwargs: Optional[dict] = None,
    num_samples: int = 21,
) -> Model:
    """
    Returns an unbuilt a multi-branch Fully Connected Neural Network.

    Parameters
    ----------
    output_size: int
        Number of target predictants.
    hidden_layers: list[int]
        List that is used to define the fully connected block. Each element creates
        a Dense layer with the corresponding units.
    n_branches: int
        (Optional) The number of branches. By default, one branch is created for each
        parameter of the probabilistic layer (or each output in the deterministic case).
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
    probabilistic_layer: str or dict
        (Optional) Name of a distribution defined in `mlpp_lib.probabilistic_layers.DISTRIBUTIONS`,
        which is used as output layer of the keras `Model`, or a dictionary `{name: options}`.
        Default is None.
    skip_connection: bool
        Include a skip connection to the MLP architecture. Default is False.
    aggregation: Literal['sum', 'concat']
        The aggregation strategy to combine the branches' outputs.
    prob_layer_kwargs: dict
        (Optional) Options of the distribution, e.g. `{"low": 0.0}` for a `TruncatedNormal`.
    num_samples: int
        Default number of samples drawn when the model outputs samples.

    Return
    ------
    model: keras model
        The unbuilt and uncompiled model.
    """

    if n_branches is None:
        if probabilistic_layer is None:
            n_branches = output_size
        else:
            n_branches = _num_distribution_parameters(
                probabilistic_layer, output_size, prob_layer_kwargs
            )

    branch_layers = [
        MultilayerPerceptron(
            hidden_layers=hidden_layers,
            batchnorm=batchnorm,
            activations=activations,
            dropout=_expand_dropout(dropout, len(hidden_layers)),
            mc_dropout=mc_dropout,
            skip_connection=skip_connection,
            indx=idx,
        )
        for idx in range(n_branches)
    ]

    mb_ffnn = MultibranchLayer(branches=branch_layers, aggregation=aggregation)

    output_layer = _build_output_layer(
        output_size,
        out_bias_init,
        probabilistic_layer,
        prob_layer_kwargs,
        num_samples,
    )

    return _assemble(mb_ffnn, output_layer, name="fully_connected_multibranch_network")


def deep_cross_network(
    output_size: int,
    hidden_layers: list,
    n_cross_layers: Optional[int] = None,
    cross_layers_hiddensize: Optional[int] = None,
    batchnorm: bool = True,
    activations: Optional[Union[str, list[str]]] = "relu",
    dropout: Optional[Union[float, list[float]]] = None,
    mc_dropout: bool = False,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]] = "zeros",
    probabilistic_layer: Optional[Union[str, dict]] = None,
    skip_connection: bool = False,
    prob_layer_kwargs: Optional[dict] = None,
    num_samples: int = 21,
) -> Model:
    """
    Build a Deep and Cross Network (see https://arxiv.org/abs/1708.05123).

    Parameters
    ----------
    output_size: int
        Number of target predictants.
    hidden_layers: list[int]
        List that is used to define the fully connected block. Each element creates
        a Dense layer with the corresponding units.
    n_cross_layers: int
        (Optional) The number of cross layers. Default is `len(hidden_layers)`.
    cross_layers_hiddensize: int
        (Optional) If given, the inputs are linearly projected to this size before
        the cross layers. By default, the cross layers act directly on the inputs.
    batchnorm: bool
        Use batch normalization. Default is True.
    activations: str or list[str]
        (Optional) Activation function(s) for the Dense layer(s). See https://keras.io/api/layers/activations/#relu-function.
        If a string is passed, the same activation is used for all layers. Default is `relu`.
    dropout: float or list[float]
        (Optional) Dropout rate for the optional dropout layers. If a `float` is passed,
        dropout layers with the given rate are created after each Dense layer.
        Default is None.
    mc_dropout: bool
        Enable Monte Carlo dropout during inference. It has no effect during training.
        It has no effect if `dropout=None`. Default is false.
    out_bias_init: str or np.ndarray
        (Optional) Specifies the initialization of the output layer bias. If a string is passed,
        it must be a valid Keras built-in initializer (see https://keras.io/api/layers/initializers/).
        If an array is passed, it must match the `output_size` argument.
    probabilistic_layer: str or dict
        (Optional) Name of a distribution defined in `mlpp_lib.probabilistic_layers.DISTRIBUTIONS`,
        which is used as output layer of the keras `Model`, or a dictionary `{name: options}`.
        Default is None.
    skip_connection: bool
        Include a skip connection to the deep part of the network. Default is False.
    prob_layer_kwargs: dict
        (Optional) Options of the distribution, e.g. `{"low": 0.0}` for a `TruncatedNormal`.
    num_samples: int
        Default number of samples drawn when the model outputs samples.

    Return
    ------
    model: keras model
        The unbuilt (and not yet compiled) model.
    """

    cross_layer = CrossNetLayer(
        hidden_size=cross_layers_hiddensize,
        depth=len(hidden_layers) if n_cross_layers is None else n_cross_layers,
    )

    deep_layer = MultilayerPerceptron(
        hidden_layers=hidden_layers,
        batchnorm=batchnorm,
        activations=activations,
        dropout=dropout,
        mc_dropout=mc_dropout,
        skip_connection=skip_connection,
    )

    encoder = ParallelConcatenateLayer([cross_layer, deep_layer])

    output_layer = _build_output_layer(
        output_size,
        out_bias_init,
        probabilistic_layer,
        prob_layer_kwargs,
        num_samples,
    )

    return _assemble(encoder, output_layer, name="deep_cross_network")
