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


# these almost work out of the box
from tensorflow_probability.python.layers import (
    IndependentNormal,
    IndependentLogistic,
    IndependentBernoulli,
    IndependentPoisson,
)

@tf.keras.saving.register_keras_serializable()
class IndependentNormal(IndependentNormal):
    @property
    def output(self):  # this is necessary to use the layer within shap
        return super().output[0]


@tf.keras.saving.register_keras_serializable()
class IndependentLogistic(IndependentLogistic):
    @property
    def output(self):
        return super().output[0]


@tf.keras.saving.register_keras_serializable()
class IndependentBernoulli(IndependentBernoulli):
    @property
    def output(self):
        return super().output[0]


@tf.keras.saving.register_keras_serializable()
class IndependentPoisson(IndependentPoisson):
    @property
    def output(self):
        return super().output[0]


@tf.keras.saving.register_keras_serializable()
class IndependentBeta(tfpl.DistributionLambda):
    """An independent 4-parameter Beta Keras layer."""

    def __init__(
        self,
        event_shape=(),
        convert_to_tensor_fn=tfd.Distribution.mean,
        validate_args=False,
        **kwargs
    ):
        """Initialize the `IndependentBeta` layer.
        Args:
        event_shape: integer vector `Tensor` representing the shape of single
            draw from this distribution.
        convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
            instance and returns a `tf.Tensor`-like object.
            Default value: `tfd.Distribution.mean`.
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

        super(IndependentBeta, self).__init__(
            lambda t: IndependentBeta.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or "IndependentBeta"):
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
            alpha, beta, shift, scale = tf.split(params, 4, axis=-1)
            # alpha > 2 and beta > 2 produce a concave downward Beta
            alpha = 2.0 + tf.math.softplus(tf.reshape(alpha, output_shape))
            beta = 2.0 + tf.math.softplus(tf.reshape(beta, output_shape))
            shift = tf.math.softplus(tf.reshape(shift, output_shape))
            scale = tf.math.softplus(tf.reshape(scale, output_shape))
            betad = tfd.Beta(alpha, beta, validate_args=validate_args)
            transf_betad = tfd.TransformedDistribution(
                distribution=betad, bijector=tfb.Shift(shift)(tfb.Scale(scale))
            )
            return independent_lib.Independent(
                transf_betad,
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args,
            )

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or "IndependentBeta_params_size"):
            event_shape = tf.convert_to_tensor(
                event_shape, name="event_shape", dtype_hint=tf.int32
            )
            return np.int32(4) * _event_size(
                event_shape, name=name or "IndependentBeta_params_size"
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
        base_config = super(IndependentBeta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output(self):
        """This allows the use of this layer with the shap package."""
        return super(IndependentBeta, self).output[0]


@tf.keras.saving.register_keras_serializable()
class IndependentGamma(tfpl.DistributionLambda):
    """An independent gamma Keras layer."""

    def __init__(
        self,
        event_shape=(),
        convert_to_tensor_fn=tfd.Distribution.mean,
        validate_args=False,
        **kwargs
    ):
        """Initialize the `IndependentGamma` layer.
        Args:
        event_shape: integer vector `Tensor` representing the shape of single
            draw from this distribution.
        convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
            instance and returns a `tf.Tensor`-like object.
            Default value: `tfd.Distribution.mean`.
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

    @property
    def output(self):
        """This allows the use of this layer with the shap package."""
        return super(IndependentGamma, self).output[0]


@tf.keras.saving.register_keras_serializable()
class IndependentLogNormal(tfpl.DistributionLambda):
    """An independent LogNormal Keras layer."""

    def __init__(
        self,
        event_shape=(),
        convert_to_tensor_fn=tfd.Distribution.mean,
        validate_args=False,
        **kwargs
    ):
        """Initialize the `IndependentLogNormal` layer.
        Args:
        event_shape: integer vector `Tensor` representing the shape of single
            draw from this distribution.
        convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
            instance and returns a `tf.Tensor`-like object.
            Default value: `tfd.Distribution.mean`.
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

        super(IndependentLogNormal, self).__init__(
            lambda t: IndependentLogNormal.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or "IndependentLogNormal"):
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
            loc, scale = tf.split(params, 2, axis=-1)
            return independent_lib.Independent(
                tfd.LogNormal(
                    loc=tf.reshape(loc, output_shape),
                    scale=tf.math.softplus(tf.reshape(scale, output_shape)) + 1e-3,
                    validate_args=validate_args,
                ),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args,
            )

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or "IndependentLogNormal_params_size"):
            event_shape = tf.convert_to_tensor(
                event_shape, name="event_shape", dtype_hint=tf.int32
            )
            return np.int32(2) * _event_size(
                event_shape, name=name or "IndependentLogNormal_params_size"
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
        base_config = super(IndependentLogNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output(self):
        """This allows the use of this layer with the shap package."""
        return super(IndependentLogNormal, self).output[0]


@tf.keras.saving.register_keras_serializable()
class IndependentTruncatedNormal(tfpl.DistributionLambda):
    """An independent TruncatedNormal Keras layer."""

    def __init__(
        self,
        event_shape=(),
        convert_to_tensor_fn=tfd.Distribution.mean,
        validate_args=False,
        **kwargs
    ):
        """Initialize the `IndependentTruncatedNormal` layer.
        Args:
        event_shape: integer vector `Tensor` representing the shape of single
            draw from this distribution.
        convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
            instance and returns a `tf.Tensor`-like object.
            Default value: `tfd.Distribution.mean`.
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

        super(IndependentTruncatedNormal, self).__init__(
            lambda t: IndependentTruncatedNormal.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or "IndependentTruncatedNormal"):
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
            loc, scale = tf.split(params, 2, axis=-1)
            return independent_lib.Independent(
                tfd.TruncatedNormal(
                    loc=tf.reshape(loc, output_shape),
                    scale=tf.math.softplus(tf.reshape(scale, output_shape)) + 1e-3,
                    low=0,
                    high=np.inf,
                    validate_args=validate_args,
                ),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args,
            )

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or "IndependentTruncatedNormal_params_size"):
            event_shape = tf.convert_to_tensor(
                event_shape, name="event_shape", dtype_hint=tf.int32
            )
            return np.int32(2) * _event_size(
                event_shape, name=name or "IndependentTruncatedNormal_params_size"
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
        base_config = super(IndependentTruncatedNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output(self):
        """This allows the use of this layer with the shap package."""
        return super(IndependentTruncatedNormal, self).output[0]


@tf.keras.saving.register_keras_serializable()
class IndependentWeibull(tfpl.DistributionLambda):
    """An independent Weibull Keras layer."""

    def __init__(
        self,
        event_shape=(),
        convert_to_tensor_fn=tfd.Distribution.mean,
        validate_args=False,
        **kwargs
    ):
        """Initialize the `IndependentWeibull` layer.
        Args:
        event_shape: integer vector `Tensor` representing the shape of single
            draw from this distribution.
        convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
            instance and returns a `tf.Tensor`-like object.
            Default value: `tfd.Distribution.mean`.
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

        super(IndependentWeibull, self).__init__(
            lambda t: IndependentWeibull.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or "IndependentWeibull"):
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
            concentration, scale = tf.split(params, 2, axis=-1)
            return independent_lib.Independent(
                tfd.Weibull(
                    concentration=tf.math.softplus(
                        tf.reshape(concentration, output_shape)
                    )
                    + 1.0,
                    scale=tf.math.softplus(tf.reshape(scale, output_shape)),
                    validate_args=validate_args,
                ),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args,
            )

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or "IndependentWeibull_params_size"):
            event_shape = tf.convert_to_tensor(
                event_shape, name="event_shape", dtype_hint=tf.int32
            )
            return np.int32(2) * _event_size(
                event_shape, name=name or "IndependentWeibull_params_size"
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
        base_config = super(IndependentWeibull, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output(self):
        """This allows the use of this layer with the shap package."""
        return super(IndependentWeibull, self).output[0]


@tf.keras.saving.register_keras_serializable()
class MultivariateNormalDiag(tfpl.DistributionLambda):
    """A `d`-variate normal Keras layer from `2* d` params,
    with a diagonal scale matrix.
    """

    def __init__(
        self,
        event_size,
        convert_to_tensor_fn=tfd.Distribution.mean,
        validate_args=False,
        **kwargs
    ):
        """Initialize the layer.
        Args:
            event_size: Scalar `int` representing the size of single draw from this
              distribution.
            convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
              instance and returns a `tf.Tensor`-like object. For examples, see
              `class` docstring.
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

        super(MultivariateNormalDiag, self).__init__(
            lambda t: MultivariateNormalDiag.new(t, event_size, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )

        self._event_size = event_size
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_size, validate_args=False, name=None):
        """Create the distribution instance from a 'params' vector."""
        with tf.name_scope(name or "MultivariateNormalDiag"):
            params = tf.convert_to_tensor(params, name="params")
            if event_size > 1:
                dist = tfd.MultivariateNormalDiag(
                    loc=params[..., :event_size],
                    scale_diag=1e-5 + tf.math.softplus(params[..., event_size:]),
                    validate_args=validate_args,
                )
            else:
                dist = tfd.Normal(
                    loc=params[..., :event_size],
                    scale=1e-5 + tf.math.softplus(params[..., event_size:]),
                    validate_args=validate_args,
                )
            return dist

    @staticmethod
    def params_size(event_size, name=None):
        """The number of 'params' needed to create a single distribution."""
        with tf.name_scope(name or "MultivariateNormalDiag_params_size"):
            return 2 * event_size

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
        base_config = super(MultivariateNormalDiag, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output(self):
        """This allows the use of this layer with the shap package."""
        return super(MultivariateNormalDiag, self).output[0]


@tf.keras.saving.register_keras_serializable()
class MultivariateNormalTriL(tfpl.MultivariateNormalTriL):
    def __init__(
        self,
        event_size,
        convert_to_tensor_fn=tfd.Distribution.mean,
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

    @property
    def output(self):
        """This allows the use of this layer with the shap package."""
        return super(MultivariateNormalTriL, self).output[0]
