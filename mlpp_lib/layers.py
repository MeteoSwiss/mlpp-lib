import logging
from typing import Literal, Optional, Union

import keras
import keras.ops as ops
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Dense,
    Dropout,
    Layer,
)

_LOGGER = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="mlpp_lib")
class MonteCarloDropout(Dropout):
    """Dropout layer that is active also at inference time."""

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


@keras.saving.register_keras_serializable(package="mlpp_lib")
class MultilayerPerceptron(Layer):
    """A fully connected layer composed of a sequence
    of linear layers interleaved by optional
    batch norms, and dropouts/MC dropouts.

    Parameters
    ----------
    hidden_layers: list[int]
        Number of units of each Dense layer.
    batchnorm: bool
        Use batch normalization after each Dense layer.
    activations: str or list[str]
        Activation function(s) for the Dense layer(s).
    dropout: float or list[float]
        Dropout rate(s). If a float is passed, a dropout layer is added after
        each Dense layer. Default is None (no dropout).
    mc_dropout: bool
        Use Monte Carlo dropout, i.e. dropout also active at inference time.
    skip_connection: bool
        Add a (linear) projection of the inputs to the output of the MLP.
    skip_connection_act: str
        Activation applied after the skip connection.
    """

    def __init__(
        self,
        hidden_layers: list,
        batchnorm: bool = False,
        activations: Optional[Union[str, list[str]]] = "relu",
        dropout: Optional[Union[float, list[float]]] = None,
        mc_dropout: bool = False,
        skip_connection: bool = False,
        skip_connection_act: str = "linear",
        indx=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(activations, list):
            assert len(activations) == len(hidden_layers)
        elif isinstance(activations, str):
            activations = [activations] * len(hidden_layers)

        if isinstance(dropout, list):
            assert len(dropout) <= len(hidden_layers)
        elif isinstance(dropout, float):
            dropout = [dropout] * (len(hidden_layers))
        else:
            if mc_dropout:
                _LOGGER.warning("dropout=None, hence I will ignore mc_dropout=True")
            dropout = []

        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.batchnorm = batchnorm
        self.activations = activations
        self.mc_dropout = mc_dropout
        self.skip_connection_act = skip_connection_act
        self.indx = indx
        self.skip_conn = skip_connection
        self.layers = []

    def build(self, input_shape):
        for i, units in enumerate(self.hidden_layers):
            d = Dense(units, name=f"dense_{self.indx}:{i}")
            d.build(input_shape)
            input_shape = (*input_shape[:-1], units)
            self.layers.append(d)
            if self.batchnorm:
                self.layers.append(BatchNormalization())
            self.layers.append(Activation(self.activations[i]))
            if i < len(self.dropout) and 0.0 < self.dropout[i] < 1.0:
                if self.mc_dropout:
                    self.layers.append(
                        MonteCarloDropout(
                            self.dropout[i], name=f"mc_dropout_{self.indx}:{i}"
                        )
                    )
                else:
                    self.layers.append(
                        Dropout(self.dropout[i], name=f"dropout_{self.indx}:{i}")
                    )

        if self.skip_conn:
            self.skip_enc = Dense(self.hidden_layers[-1], name="skip_dense")
            self.skip_add = Add(name="skip_add")
            self.skip_act = Activation(
                activation=self.skip_connection_act, name="skip_activation"
            )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.hidden_layers[-1])

    def call(self, inputs, training=None):
        out = inputs
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                out = layer(out, training=training)
            else:
                out = layer(out)
        # optional skip connection
        if self.skip_conn:
            out = self.skip_add([out, self.skip_enc(inputs)])
            out = self.skip_act(out)
        return out


@keras.saving.register_keras_serializable(package="mlpp_lib")
class MultibranchLayer(Layer):
    """Feeds the same input to all branches and aggregates their outputs."""

    def __init__(
        self,
        branches: list[Layer],
        aggregation: Literal["sum", "concat"] = "concat",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.branches = branches
        self.aggregation_type = aggregation
        self.aggr = (
            keras.layers.Concatenate(axis=-1)
            if aggregation == "concat"
            else keras.layers.Add()
        )

    def call(self, inputs, training=None):
        branch_outputs = [branch(inputs, training=training) for branch in self.branches]
        return self.aggr(branch_outputs)

    def compute_output_shape(self, input_shape):
        shapes = [branch.compute_output_shape(input_shape) for branch in self.branches]
        if self.aggregation_type == "concat":
            return (*shapes[0][:-1], sum(shape[-1] for shape in shapes))
        return shapes[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "branches": [
                    keras.layers.serialize(branch) for branch in self.branches
                ],
                "aggregation": self.aggregation_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Recreates the layer from its config."""
        branches = [
            keras.layers.deserialize(branch_config)
            for branch_config in config.pop("branches")
        ]
        return cls(branches=branches, **config)


@keras.saving.register_keras_serializable(package="mlpp_lib")
class CrossNetLayer(Layer):
    """Cross network of a Deep & Cross Network (https://arxiv.org/abs/1708.05123).

    x_{l+1} = x_0 * (x_l^T w_l) + b_l + x_l

    Parameters
    ----------
    hidden_size: int
        If given, the inputs are first linearly projected to `hidden_size`
        dimensions. Otherwise, the cross layers act directly on the inputs.
    depth: int
        Number of cross layers.
    """

    def __init__(self, hidden_size: Optional[int] = None, depth: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.depth = depth

    def build(self, input_shape):
        if self.hidden_size is not None:
            self.encoder = Dense(self.hidden_size, name="cross_encoder")
            self.encoder.build(input_shape)
            size = self.hidden_size
        else:
            self.encoder = None
            size = input_shape[-1]
        self.ws = [
            self.add_weight(
                shape=(size, 1), initializer="glorot_uniform", name=f"w_{d}"
            )
            for d in range(self.depth)
        ]
        self.bs = [
            self.add_weight(shape=(size,), initializer="zeros", name=f"b_{d}")
            for d in range(self.depth)
        ]
        super().build(input_shape)

    def call(self, x):
        x0 = self.encoder(x) if self.encoder is not None else x
        x_l = x0
        for w, b in zip(self.ws, self.bs):
            # x0 * x_l^T w is the (cheaper) equivalent of (x0 x_l^T) w
            x_l = x0 * ops.matmul(x_l, w) + b + x_l
        return x_l

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.hidden_size or input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "depth": self.depth})
        return config


@keras.saving.register_keras_serializable(package="mlpp_lib")
class ParallelConcatenateLayer(Layer):
    """Feeds the same input to all given layers
    and concatenates their outputs along the last dimension.
    """

    def __init__(self, layers: list[Layer], **kwargs):
        super().__init__(**kwargs)
        self.layers = layers
        self.concat = keras.layers.Concatenate(axis=-1)

    def call(self, inputs, training=None):
        return self.concat([layer(inputs, training=training) for layer in self.layers])

    def compute_output_shape(self, input_shape):
        shapes = [layer.compute_output_shape(input_shape) for layer in self.layers]
        return (*shapes[0][:-1], sum(shape[-1] for shape in shapes))

    def get_config(self):
        config = super().get_config()
        config.update(
            {"layers": [keras.layers.serialize(layer) for layer in self.layers]}
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Recreates the layer from its config."""
        layers = [
            keras.layers.deserialize(layer_config)
            for layer_config in config.pop("layers")
        ]
        return cls(layers=layers, **config)
