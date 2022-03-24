"""In this module, any custom built keras layers are included."""
import numpy as np
import tensorflow as tf
import tensorflow_probability.python.layers as tfpl
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers.distribution_layer import (
    _event_size,
    _get_convert_to_tensor_fn,
    _serialize,
    dist_util,
    independent_lib,
)


# these work out of the box
from tensorflow_probability.python.layers import (
    IndependentNormal,
    IndependentLogistic,
    IndependentBernoulli,
    IndependentPoisson,
)


@tf.keras.utils.register_keras_serializable()
class MultivariateNormalTriL(tfpl.MultivariateNormalTriL):
    def __init__(
        self,
        event_size,
        convert_to_tensor_fn=tfd.Distribution.sample,
        validate_args=False,
        **kwargs
    ):
        convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

        # If there is a 'make_distribution_fn' keyword argument (e.g., because we
        # are being called from a `from_config` method), remove it.  We pass the
        # distribution function to `DistributionLambda.__init__` below as the first
        # positional argument.
        kwargs.pop("make_distribution_fn", None)

        super().__init__(event_size, convert_to_tensor_fn, validate_args, **kwargs)
        self._event_size = event_size
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    def get_config(self):
        """Returns the config of this layer.
        NOTE: At the moment, this configuration can only be serialized if the
        Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
        implements `get_config`) or one of the standard values:
        - `Distribution.sample` (or `"sample"`)
        - `Distribution.mean` (or `"mean"`)
        - `Distribution.mode` (or `"mode"`)
        - `Distribution.stddev` (or `"stddev"`)
        - `Distribution.variance` (or `"variance"`)
        """
        config = {
            "event_size": self._event_size,
            "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
            "validate_args": self._validate_args,
        }
        base_config = super(MultivariateNormalTriL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable()
class IndependentGamma(tfpl.DistributionLambda):
    """An independent gamma Keras layer."""

    def __init__(
        self,
        event_shape=(),
        convert_to_tensor_fn=tfd.Distribution.sample,
        validate_args=False,
        **kwargs
    ):
        """Initialize the `IndependentGamma` layer.
        Args:
        event_shape: integer vector `Tensor` representing the shape of single
            draw from this distribution.
        convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
            instance and returns a `tf.Tensor`-like object.
            Default value: `tfd.Distribution.sample`.
        validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
            Default value: `False`.
        **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
        """
        convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

        # If there is a 'make_distribution_fn' keyword argument (e.g., because we
        # are being called from a `from_config` method), remove it.  We pass the
        # distribution function to `DistributionLambda.__init__` below as the first
        # positional argument.
        kwargs.pop("make_distribution_fn", None)

        super(IndependentGamma, self).__init__(
            lambda t: IndependentGamma.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or "IndependentGamma"):
            params = tf.convert_to_tensor(params, name="params")
            event_shape = dist_util.expand_to_vector(
                tf.convert_to_tensor(
                    event_shape, name="event_shape", dtype_hint=tf.int32
                ),
                tensor_name="event_shape",
            )
            output_shape = tf.concat(
                [
                    tf.shape(params)[:-1],
                    event_shape,
                ],
                axis=0,
            )
            concentration, rate = tf.split(params, 2, axis=-1)
            return independent_lib.Independent(
                tfd.Gamma(
                    concentration=tf.math.softplus(
                        tf.reshape(concentration, output_shape)
                    ),
                    rate=tf.math.softplus(tf.reshape(rate, output_shape)),
                    validate_args=validate_args,
                ),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args,
            )

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or "IndependentGamma_params_size"):
            event_shape = tf.convert_to_tensor(
                event_shape, name="event_shape", dtype_hint=tf.int32
            )
            return np.int32(2) * _event_size(
                event_shape, name=name or "IndependentGamma_params_size"
            )

    def get_config(self):
        """Returns the config of this layer.
        NOTE: At the moment, this configuration can only be serialized if the
        Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
        implements `get_config`) or one of the standard values:
        - `Distribution.sample` (or `"sample"`)
        - `Distribution.mean` (or `"mean"`)
        - `Distribution.mode` (or `"mode"`)
        - `Distribution.stddev` (or `"stddev"`)
        - `Distribution.variance` (or `"variance"`)
        """
        config = {
            "event_shape": self._event_shape,
            "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
            "validate_args": self._validate_args,
        }
        base_config = super(IndependentGamma, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
