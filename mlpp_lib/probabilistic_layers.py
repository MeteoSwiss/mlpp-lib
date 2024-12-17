import torch
import torch.nn as nn
import keras
from keras.layers import TorchModuleWrapper, Dense
from keras import Layer
from abc import ABC, abstractmethod
import numpy as np
from keras import initializers
from typing import Literal
from inspect import getmembers, isclass
import sys

from mlpp_lib.custom_distributions import TruncatedNormalDistribution, CensoredNormalDistribution
from mlpp_lib.layers import MeanAndTriLCovLayer

class BaseParametricDistributionModule(ABC):
    """ Base class for parametric distributions layers
    """
    @property
    @abstractmethod
    def num_parameters(self):
        '''The number of parameters that describe the distribution'''
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    def has_rsample(self):
        return self._distribution.has_rsample
    
class UniveriateGaussianModule(nn.Module, BaseParametricDistributionModule):
    '''
    Torch implementation of a Gaussian sampling layer given mean and covariance
    values of shape [None, 2]. This layer uses the reparametrization trick
    to allow the flow of gradients.
    '''
    _name = 'gaussian'
    _distribution = torch.distributions.Normal
    def __init__(self, **kwargs):
        super(UniveriateGaussianModule, self).__init__()
        self.get_positive_std = torch.nn.Softplus()

    def forward(self, moments, num_samples=1, return_dist=False):
        
        # Create a copy of `moments` to avoid issues when using Softplus 
        # on tensor selections 
        new_moments = moments.clone()  
        new_moments[:, 1] = self.get_positive_std(moments[:, 1])

        
        normal_dist = self._distribution(new_moments[:,0:1], new_moments[:,1:2])
        if return_dist:
            return normal_dist
        samples = normal_dist.rsample(sample_shape=(num_samples,))
        return samples.permute(1,0,2)

    
    @property
    def num_parameters(self):
        return 2
    
    @property
    def name(self):
        return self._name
    
class MultivariateGaussianTriLModule(nn.Module, BaseParametricDistributionModule):
    """Multivariate Gaussian ~N(mu, L) where mu = E[x] is the mean vector, and L is a lower triangular 
    matrix such that LL^T = Cov(x). Matrix L is only required to be a lower triagular square matrix. 
    Internally, the values on the diagonal will be ensured positive with a softplus. 
    """
    _name = 'multivariate_tril_gaussian'
    _distribution = torch.distributions.MultivariateNormal
    
    def __init__(self, dim, **kwargs):
        super(MultivariateGaussianTriLModule, self).__init__()
        self.dim = dim
        
    def forward(self, mean_and_tril_cov, num_samples=1, return_dist=True):
        mean = mean_and_tril_cov[0]
        tril_cov = mean_and_tril_cov[1]
        
        tril_cov = self._ensure_lower_cholesky(tril_cov)
        
        multivariate_normal = self._distribution(loc=mean, scale_tril=tril_cov)
        if return_dist:
            return multivariate_normal
        samples = multivariate_normal.rsample(sample_shape=(num_samples,))
        return samples.permute(1,0,2)
        
        
    def _ensure_lower_cholesky(self, x):
        """Ensures positive values on the diagonal. 
        The input is expected to be a lower triangular matrix.

        """
        diag = torch.diagonal(x, dim1=-2, dim2=-1) # get diagonals
        diag_fixed = torch.nn.functional.softplus(diag) # make them positive
        # remove old diagonals and replace the new ones
        return x - torch.diag_embed(diag) + torch.diag_embed(diag_fixed)
    
    @property
    def num_parameters(self):
        return (self.dim, self.dim * (self.dim + 1) // 2)
    
    @property
    def name(self):
        return self._name
    
class UnivariateTruncatedGaussianModule(nn.Module, BaseParametricDistributionModule):
    _name = 'truncated_gaussian'
    _distribution = TruncatedNormalDistribution
    
    def __init__(self, a, b, **kwargs):
        super(UnivariateTruncatedGaussianModule, self).__init__()
        self.get_positive_std = torch.nn.Softplus()
        if type(a) != torch.Tensor:
            a = torch.tensor(a)
        if type(b) != torch.Tensor:
            b = torch.tensor(b)
            
        self.a, self.b = a, b
            
        
    def forward(self, moments, num_samples=1, return_dist=False):
        
        # Create a copy of `moments` to avoid issues when using Softplus 
        # on tensor selections 
        new_moments = moments.clone()  
        new_moments[:, 1] = self.get_positive_std(moments[:, 1])

        
        trunc_normal_dist = self._distribution(mu_bar=new_moments[:,0:1], sigma_bar=new_moments[:,1:2], a=self.a, b=self.b)
        if return_dist:
            return trunc_normal_dist
        samples = trunc_normal_dist.rsample(sample_shape=(num_samples,))
        return samples.permute(1,0,2)

    @property
    def num_parameters(self):
        return 2
    
    @property
    def name(self):
        return self._name
    
class UnivariateCensoredGaussianModule(nn.Module, BaseParametricDistributionModule):
    _name = 'censored_gaussian'
    _distribution = CensoredNormalDistribution
    def __init__(self, a: torch.Tensor,b: torch.Tensor, **kwargs):
        super(UnivariateCensoredGaussianModule, self).__init__()
        self.get_positive_std = torch.nn.Softplus()
        if type(a) != torch.Tensor:
            a = torch.tensor(a)
        if type(b) != torch.Tensor:
            b = torch.tensor(b)
        self.a, self.b = a, b
        
    def forward(self, moments, num_samples=1, return_dist=False):
        new_moments = moments.clone()  
        new_moments[:, 1] = self.get_positive_std(moments[:, 1])
        
        censored_normal_dist = self._distribution(mu_bar=new_moments[:,0:1], sigma_bar=new_moments[:,1:2], a=self.a, b=self.b)
        if return_dist:
            return censored_normal_dist
        samples = censored_normal_dist.sample((num_samples,))
        
        return samples.permute(1,0,2)

    @property
    def num_parameters(self):
        return 2
    
    @property
    def name(self):
        return self._name
    
class UnivariateLogNormalModule(nn.Module, BaseParametricDistributionModule):
    """
    Module implementing Y such that
    X ~ Normal(loc, scale)
    Y = exp(X) ~ LogNormal(loc, scale)
    """
    _name = 'log_gaussian'
    _distribution = torch.distributions.LogNormal

    def __init__(self, **kwargs):
        super(UnivariateLogNormalModule, self).__init__()
        self.get_positive_std = torch.nn.Softplus()
        
        
    def forward(self, moments, num_samples=1, return_dist=False):
        new_moments = moments.clone()  
        new_moments[:, 1] = self.get_positive_std(moments[:, 1])
        
        normal_dist = self._distribution(new_moments[:,0:1], new_moments[:,1:2])
        if return_dist:
            return normal_dist
        samples = normal_dist.rsample(sample_shape=(num_samples,))
        return samples.permute(1,0,2)
        
    @property
    def num_parameters(self):
        return 2
    
    @property
    def name(self):
        return self._name
    
class WeibullModule(nn.Module, BaseParametricDistributionModule):
    """
    Toch implementation of a 2-parameters Weibull distribution.
    """
    _name = 'weibull'
    _distribution = torch.distributions.Weibull
    def __init__(self, **kwargs):
        super(WeibullModule, self).__init__()
        self.get_positive_params = torch.nn.Softplus()
        
    def forward(self, params, num_samples=1, return_dist=False):
        params = self.get_positive_params(params)

        weibull_dist = self._distribution(scale=params[:,0:1],
                                                   concentration=params[:,1:2])
        if return_dist:
            return weibull_dist
        
        samples = weibull_dist.rsample(sample_shape=(num_samples,))

        return samples.permute(1,0,2)
        return samples
    
    @property
    def num_parameters(self):
        return 2
    
    @property
    def name(self):
        return self._name
    
    
class ExponentialModule(nn.Module, BaseParametricDistributionModule):
    _name = 'exponential'
    _distribution = torch.distributions.Exponential
    def __init__(self, **kwargs):
        super(ExponentialModule, self).__init__()
        self.get_positive_lambda = torch.nn.Softplus()
    
    def forward(self, params, num_samples=1, return_dist=False):
        params = self.get_positive_lambda(params)

        # return torch.distributions.Exponential(rate=params)
        exp_dist = self._distribution(rate=params)
        if return_dist:
            return exp_dist
        samples = exp_dist.rsample(sample_shape=(num_samples,))

        return samples.permute(1,0,2)
    
    @property
    def num_parameters(self):
        return 1
    
    @property
    def name(self):
        return self._name
    
class BetaModule(nn.Module, BaseParametricDistributionModule):
    _name = 'beta'
    _distribution = torch.distributions.Beta
    
    def __init__(self, **kwargs):
        super(BetaModule, self).__init__()
        
        self.get_positive_concentrations = torch.nn.Softplus()
        
    def forward(self, params, num_samples=1, return_dist=False):
        params = self.get_positive_concentrations(params)
        
        beta_dist = self._distribution(concentration1=params[:,0:1], # c1 = alpha
                                             concentration0=params[:,1:2]) # c0 = beta
        
        if return_dist:
            return beta_dist

        samples = beta_dist.rsample(sample_shape=(num_samples,))
        return samples.permute(1,0,2)
        return samples
    
    @property
    def num_parameters(self):
        return 2
    
    @property
    def name(self):
        return self._name
    
    
class GammaModule(nn.Module, BaseParametricDistributionModule):
    
    _name = 'gamma'
    _distribution = torch.distributions.Gamma
    def __init__(self, **kwargs):
        super(GammaModule, self).__init__()
        self.get_positive_params = torch.nn.Softplus()
        
    def forward(self, params, num_samples=1, return_dist=False):
        params = self.get_positive_params(params)
        
        gamma_dist = self._distribution(concentration=params[:,0:1], # alpha
                                               rate=params[:,1:2]) # beta or 1/scale
        
        if return_dist:
            return gamma_dist

        samples = gamma_dist.rsample(sample_shape=(num_samples,))
        return samples.permute(1,0,2)
        return samples
    
    @property
    def num_parameters(self):
        return 2
    
    @property
    def name(self):
        return self._name

@keras.saving.register_keras_serializable()
class BaseDistributionLayer(Layer):
    '''
    Keras layer implementing a sampling layer with reparametrization
    trick, based on an underlying parametric distribution.
    This layer is responsible of preparing the shape of the input to the probabilistic layer 
    to match the number of parameters. It does not assume anything about their properties as 
    it merely applies a linear layer. The underlying probabilistic layer needs to take care 
    of parameter constraints, e.g, the positiveness of the parameters.
    '''
    def __init__(self, distribution: BaseParametricDistributionModule, 
                 num_samples: int=21, 
                 bias_init = 'zeros',
                 **kwargs):
        super(BaseDistributionLayer, self).__init__(**kwargs)

        self.prob_layer = TorchModuleWrapper(distribution, name=distribution.name)
        self.num_dist_params = distribution.num_parameters
        
        if isinstance(bias_init, np.ndarray):
            bias_init = np.hstack(
                [bias_init, [0.0] * (distribution.num_parameters - bias_init.shape[0])]
            )
            bias_init = initializers.Constant(bias_init)
        self.bias_init = bias_init
        # linear layer to map any input size into the number of parameters of the underlying distribution.
        self.num_samples=num_samples
        self.is_multivariate_gaussian = isinstance(distribution, MultivariateGaussianTriLModule)

    def build(self, input_shape):
        if not self.is_multivariate_gaussian:
            self.parameters_encoder = Dense(self.num_dist_params, name='parameters_encoder', bias_initializer=self.bias_init)
        else:
            self.parameters_encoder = MeanAndTriLCovLayer(d1=self.prob_layer.module.num_parameters[0])
        super().build(input_shape)

    def call(self, inputs, output_type: Literal["distribution", "samples"]='distribution', training=None, **kwargs):
        predicted_parameters = self.parameters_encoder(inputs)
        if output_type == 'distribution':
            dist = self.prob_layer(predicted_parameters, num_samples=0, return_dist=True)
            return dist
        elif output_type == 'samples':
            if training and not self.prob_layer.module.has_rsample:
                raise MissingReparameterizationError(f"Gradient-based optimization will not work, as the underlying {self.prob_layer.module._distribution.__name__} distribution does not have a reparametrized sampling function.")
            samples = self.prob_layer(predicted_parameters, num_samples=self.num_samples, return_dist=False)
            return samples

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_samples, 1)

    def get_config(self):
        config = super(BaseDistributionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    # @property
    # def scoringrules_param_order(self):
    #     key = self.prob_layer.name
        
    #     _param_orders = {
    #         'univariate_gaussian': ['loc', 'scale'],
    #         'exponential': ['rate']
    #     }
        
    #     return _param_orders[key]
        
class MissingReparameterizationError(Exception):
    """Raised when a sampling function without 'rsample' is used in a context requiring reparameterization."""
    pass      
    
    
    
distribution_to_layer = {obj[1]._name: obj[1] for obj in getmembers(sys.modules[__name__], isclass) 
                 if issubclass(obj[1], BaseParametricDistributionModule) and obj[0] != 'BaseParametricDistributionModule'}





# """In this module, any custom built keras layers are included."""

# import numpy as np
# import tensorflow as tf
# import tensorflow_probability.python.layers as tfpl
# from tensorflow_probability.python import bijectors as tfb
# from tensorflow_probability.python import distributions as tfd
# from tensorflow_probability.python.layers.distribution_layer import (
#     _event_size,
#     _get_convert_to_tensor_fn,
#     _serialize,
#     dist_util,
#     independent_lib,
# )


# # these almost work out of the box
# from tensorflow_probability.python.layers import (
#     IndependentNormal,
#     IndependentLogistic,
#     IndependentBernoulli,
#     IndependentPoisson,
# )


# @tf.keras.saving.register_keras_serializable()
# class IndependentNormal(IndependentNormal):
#     @property
#     def output(self):  # this is necessary to use the layer within shap
#         return super().output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentLogistic(IndependentLogistic):
#     @property
#     def output(self):
#         return super().output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentBernoulli(IndependentBernoulli):
#     @property
#     def output(self):
#         return super().output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentPoisson(IndependentPoisson):
#     @property
#     def output(self):
#         return super().output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentBeta(tfpl.DistributionLambda):
#     """An independent 2-parameter Beta Keras layer"""

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentBeta` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return IndependentBeta.new(t, event_shape, validate_args)

#         super(IndependentBeta, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentBeta"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             alpha, beta = tf.split(params, 2, axis=-1)

#             alpha = tf.math.softplus(tf.reshape(alpha, output_shape)) + 1e-3
#             beta = tf.math.softplus(tf.reshape(beta, output_shape)) + 1e-3
#             betad = tfd.Beta(alpha, beta, validate_args=validate_args)

#             return independent_lib.Independent(
#                 betad,
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentBeta_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(2) * _event_size(
#                 event_shape, name=name or "IndependentBeta_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentBeta, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentBeta, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class Independent4ParamsBeta(tfpl.DistributionLambda):
#     """An independent 4-parameter Beta Keras layer allowing control over scale as well as a 'shift' parameter."""

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `Independent4ParamsBeta` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return Independent4ParamsBeta.new(t, event_shape, validate_args)

#         super(Independent4ParamsBeta, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "Independent4ParamsBeta"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             alpha, beta, shift, scale = tf.split(params, 4, axis=-1)
#             # alpha > 2 and beta > 2 produce a concave downward Beta
#             alpha = tf.math.softplus(tf.reshape(alpha, output_shape)) + 1e-3
#             beta = tf.math.softplus(tf.reshape(beta, output_shape)) + 1e-3
#             shift = tf.math.softplus(tf.reshape(shift, output_shape))
#             scale = tf.math.softplus(tf.reshape(scale, output_shape)) + 1e-3
#             betad = tfd.Beta(alpha, beta, validate_args=validate_args)
#             transf_betad = tfd.TransformedDistribution(
#                 distribution=betad, bijector=tfb.Shift(shift)(tfb.Scale(scale))
#             )
#             return independent_lib.Independent(
#                 transf_betad,
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "Independent4ParamsBeta_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(4) * _event_size(
#                 event_shape, name=name or "Independent4ParamsBeta_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(Independent4ParamsBeta, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(Independent4ParamsBeta, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentDoublyCensoredNormal(tfpl.DistributionLambda):
#     """An independent censored normal Keras layer."""

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentDoublyCensoredNormal` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)
#         # get the clipping parameters and pop them
#         _clip_low = kwargs.pop("clip_low", 0.0)
#         _clip_high = kwargs.pop("clip_high", 1.0)

#         def new_from_t(t):
#             return IndependentDoublyCensoredNormal.new(
#                 t,
#                 event_shape,
#                 validate_args,
#                 clip_low=_clip_low,
#                 clip_high=_clip_high,
#             )

#         super(IndependentDoublyCensoredNormal, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(
#         params,
#         event_shape=(),
#         validate_args=False,
#         name=None,
#         clip_low=0.0,
#         clip_high=1.0,
#     ):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentDoublyCensoredNormal"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             loc, scale = tf.split(params, 2, axis=-1)
#             loc = tf.reshape(loc, output_shape)
#             scale = tf.math.softplus(tf.reshape(scale, output_shape)) + 1e-3
#             normal_dist = tfd.Normal(loc=loc, scale=scale, validate_args=validate_args)

#             class CustomCensored(tfd.Distribution):
#                 def __init__(self, normal, clip_low=0.0, clip_high=1.0):
#                     self.normal = normal
#                     super(CustomCensored, self).__init__(
#                         dtype=normal.dtype,
#                         reparameterization_type=tfd.FULLY_REPARAMETERIZED,
#                         validate_args=validate_args,
#                         allow_nan_stats=True,
#                     )
#                     self.clip_low = clip_low
#                     self.clip_high = clip_high

#                 def _sample_n(self, n, seed=None):

#                     # Sample from normal distribution
#                     samples = self.normal.sample(sample_shape=(n,), seed=seed)

#                     # Clip values between 0 and 1
#                     chosen_samples = tf.clip_by_value(
#                         samples, self.clip_low, self.clip_high
#                     )

#                     return chosen_samples

#                 def _mean(self):
#                     """
#                     Original: X ~ N(mu, sigma)
#                     Censored: Y = X if clip_low <= X <= clip_high else clip_low if X < clip_low else clip_high
#                     Phi / phi: CDF / PDF of standard normal distribution

#                     Law of total expectations:
#                         E[Y] = E[Y | X > c_h] * P(X > c_h) + E[Y | X < c_l] * P(X < c_l) + E[Y | c_l <= X <= c_h] * P(c_l <= X <= c_h)
#                              = c_h * P(X > c_h) + P(X < c_l) * c_l + E[Y | c_l <= X <= c_h] * P(c_l <= X <= c_h)
#                              = c_h * P(X > c_h) + P(X < c_l) * c_l + E[Z ~ TruncNormal(mu, sigma, c_l, c_h)] * (Phi((c_h - mu) / sigma) - Phi(c_l - mu / sigma))
#                              = c_h * (1 - Phi((c_h - mu) / sigma))
#                                 + c_l * Phi((c_l - mu) / sigma)
#                                 + mu * (Phi((c_h - mu) / sigma) - Phi(c_l - mu / sigma))
#                                 + sigma * (phi(c_l - mu / sigma) - phi((c_h - mu) / sigma))
#                     Ref for TruncatedNormal mean: https://en.wikipedia.org/wiki/Truncated_normal_distribution
#                     """
#                     mu, sigma = self.normal.mean(), self.normal.stddev()
#                     low_bound_standard = (self.clip_low - mu) / sigma
#                     high_bound_standard = (self.clip_high - mu) / sigma

#                     cdf = lambda x: tfd.Normal(0, 1).cdf(x)
#                     pdf = lambda x: tfd.Normal(0, 1).prob(x)

#                     return (
#                         self.clip_high * (1 - cdf(high_bound_standard))
#                         + self.clip_low * cdf(low_bound_standard)
#                         + mu * (cdf(high_bound_standard) - cdf(low_bound_standard))
#                         + sigma * (pdf(low_bound_standard) - pdf(high_bound_standard))
#                     )

#                 def _log_prob(self, value):

#                     mu, sigma = self.normal.mean(), self.normal.stddev()
#                     cdf = lambda x: tfd.Normal(0, 1).cdf(x)
#                     pdf = lambda x: tfd.Normal(0, 1).prob(x)

#                     logprob_left = lambda x: tf.math.log(
#                         cdf(self.clip_low - mu / sigma) + 1e-3
#                     )
#                     logprob_middle = lambda x: self.normal.log_prob(x)
#                     logprob_right = lambda x: tf.math.log(
#                         1 - cdf((self.clip_high - mu) / sigma) + 1e-3
#                     )

#                     return (
#                         logprob_left(value)
#                         + logprob_middle(value)
#                         + logprob_right(value)
#                     )

#             return independent_lib.Independent(
#                 CustomCensored(normal_dist, clip_low=clip_low, clip_high=clip_high),
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentDoublyCensoredNormal_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(2) * _event_size(
#                 event_shape, name=name or "IndependentDoublyCensoredNormal_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentDoublyCensoredNormal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentDoublyCensoredNormal, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentConcaveBeta(tfpl.DistributionLambda):
#     """An independent 4-parameter Beta Keras layer with enforced concavity"""

#     # INdependent
#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentConcaveBeta` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return IndependentConcaveBeta.new(t, event_shape, validate_args)

#         super(IndependentConcaveBeta, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentConcaveBeta"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             alpha, beta, shift, scale = tf.split(params, 4, axis=-1)
#             # alpha > 2 and beta > 2 produce a concave downward Beta
#             alpha = tf.math.softplus(tf.reshape(alpha, output_shape)) + 2.0
#             beta = tf.math.softplus(tf.reshape(beta, output_shape)) + 2.0
#             shift = tf.math.softplus(tf.reshape(shift, output_shape))
#             scale = tf.math.softplus(tf.reshape(scale, output_shape)) + 1e-3
#             betad = tfd.Beta(alpha, beta, validate_args=validate_args)
#             transf_betad = tfd.TransformedDistribution(
#                 distribution=betad, bijector=tfb.Shift(shift)(tfb.Scale(scale))
#             )
#             return independent_lib.Independent(
#                 transf_betad,
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentConcaveBeta_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(4) * _event_size(
#                 event_shape, name=name or "IndependentConcaveBeta_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentConcaveBeta, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentConcaveBeta, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentGamma(tfpl.DistributionLambda):
#     """An independent gamma Keras layer."""

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentGamma` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return IndependentGamma.new(t, event_shape, validate_args)

#         super(IndependentGamma, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentGamma"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             concentration, rate = tf.split(params, 2, axis=-1)
#             return independent_lib.Independent(
#                 tfd.Gamma(
#                     concentration=tf.math.softplus(
#                         tf.reshape(concentration, output_shape)
#                     ),
#                     rate=tf.math.softplus(tf.reshape(rate, output_shape)),
#                     validate_args=validate_args,
#                 ),
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentGamma_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(2) * _event_size(
#                 event_shape, name=name or "IndependentGamma_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentGamma, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentGamma, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentLogNormal(tfpl.DistributionLambda):
#     """An independent LogNormal Keras layer."""

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentLogNormal` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return IndependentLogNormal.new(t, event_shape, validate_args)

#         super(IndependentLogNormal, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentLogNormal"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             loc, scale = tf.split(params, 2, axis=-1)
#             return independent_lib.Independent(
#                 tfd.LogNormal(
#                     loc=tf.reshape(loc, output_shape),
#                     scale=tf.math.softplus(tf.reshape(scale, output_shape)) + 1e-3,
#                     validate_args=validate_args,
#                 ),
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentLogNormal_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(2) * _event_size(
#                 event_shape, name=name or "IndependentLogNormal_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentLogNormal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentLogNormal, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentLogitNormal(tfpl.DistributionLambda):
#     """An independent Logit-Normal Keras layer."""

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.sample,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentLogitNormal` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return IndependentLogitNormal.new(t, event_shape, validate_args)

#         super(IndependentLogitNormal, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentLogitNormal"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             loc, scale = tf.split(params, 2, axis=-1)
#             return independent_lib.Independent(
#                 tfd.LogitNormal(
#                     loc=tf.reshape(loc, output_shape),
#                     scale=tf.math.softplus(tf.reshape(scale, output_shape)) + 1e-3,
#                     validate_args=validate_args,
#                 ),
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentLogitNormal_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(2) * _event_size(
#                 event_shape, name=name or "IndependentLogitNormal_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentLogitNormal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentLogitNormal, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentMixtureNormal(tfpl.DistributionLambda):
#     """A mixture of two normal distributions Keras layer.
#     5-parameters distribution: loc1, scale1, loc2, scale2, weight
#     """

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentMixtureNormal` layer.
#         Args:
#             event_shape: integer vector `Tensor` representing the shape of single
#                 draw from this distribution.
#             convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#                 instance and returns a `tf.Tensor`-like object.
#                 Default value: `tfd.Distribution.mean`.
#             validate_args: Python `bool`, default `False`. When `True` distribution
#                 parameters are checked for validity despite possibly degrading runtime
#                 performance. When `False` invalid inputs may silently render incorrect
#                 outputs.
#                 Default value: `False`.
#             **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """

#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return IndependentMixtureNormal.new(t, event_shape, validate_args)

#         super(IndependentMixtureNormal, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentMixtureNormal"):
#             params = tf.convert_to_tensor(params, name="params")

#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )

#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )

#             loc1, scale1, loc2, scale2, weight = tf.split(params, 5, axis=-1)
#             loc1 = tf.reshape(loc1, output_shape)
#             scale1 = tf.math.softplus(tf.reshape(scale1, output_shape)) + 1e-3
#             loc2 = tf.reshape(loc2, output_shape)
#             scale2 = tf.math.softplus(tf.reshape(scale2, output_shape)) + 1e-3
#             weight = tf.math.sigmoid(tf.reshape(weight, output_shape))

#             # Create the component distributions
#             normald1 = tfd.Normal(loc=loc1, scale=scale1)
#             normald2 = tfd.Normal(loc=loc2, scale=scale2)

#             # Create a categorical distribution for the weights
#             cat = tfd.Categorical(
#                 probs=tf.concat(
#                     [tf.expand_dims(weight, -1), tf.expand_dims(1 - weight, -1)],
#                     axis=-1,
#                 )
#             )

#             class CustomMixture(tfd.Distribution):
#                 def __init__(self, cat, normald1, normald2):
#                     self.cat = cat
#                     self.normald1 = normald1
#                     self.normald2 = normald2
#                     super(CustomMixture, self).__init__(
#                         dtype=normald1.dtype,
#                         reparameterization_type=tfd.FULLY_REPARAMETERIZED,
#                         validate_args=validate_args,
#                         allow_nan_stats=True,
#                     )

#                 def _sample_n(self, n, seed=None):
#                     indices = self.cat.sample(sample_shape=(n,), seed=seed)

#                     # Sample from both truncated normal distributions
#                     samples1 = self.normald1.sample(sample_shape=(n,), seed=seed)
#                     samples2 = self.normald2.sample(sample_shape=(n,), seed=seed)

#                     # Stack the samples along a new axis
#                     samples = tf.stack([samples1, samples2], axis=-1)

#                     # Gather samples according to indices from the categorical distribution
#                     chosen_samples = tf.gather(
#                         samples,
#                         indices,
#                         batch_dims=tf.get_static_value(tf.rank(indices)),
#                     )

#                     return chosen_samples

#                 def _log_prob(self, value):
#                     log_prob1 = self.normald1.log_prob(value)
#                     log_prob2 = self.normald2.log_prob(value)
#                     log_probs = tf.stack([log_prob1, log_prob2], axis=-1)
#                     weighted_log_probs = log_probs + tf.math.log(
#                         tf.concat([weight, 1 - weight], axis=-1)
#                     )
#                     return tf.reduce_logsumexp(weighted_log_probs, axis=-1)

#                 def _mean(self):
#                     return (
#                         weight * self.normald1.mean()
#                         + (1 - weight) * self.normald2.mean()
#                     )

#             mixtured = CustomMixture(cat, normald1, normald2)

#             return independent_lib.Independent(
#                 mixtured,
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentMixtureNormal_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(5) * _event_size(
#                 event_shape, name=name or "IndependentMixtureNormal_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentMixtureNormal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentMixtureNormal, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentTruncatedNormal(tfpl.DistributionLambda):
#     """An independent TruncatedNormal Keras layer."""

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentTruncatedNormal` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return IndependentTruncatedNormal.new(t, event_shape, validate_args)

#         super(IndependentTruncatedNormal, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentTruncatedNormal"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             loc, scale = tf.split(params, 2, axis=-1)
#             return independent_lib.Independent(
#                 tfd.TruncatedNormal(
#                     loc=tf.reshape(loc, output_shape),
#                     scale=tf.math.softplus(tf.reshape(scale, output_shape)) + 1e-3,
#                     low=0,
#                     high=np.inf,
#                     validate_args=validate_args,
#                 ),
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentTruncatedNormal_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(2) * _event_size(
#                 event_shape, name=name or "IndependentTruncatedNormal_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentTruncatedNormal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentTruncatedNormal, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class IndependentWeibull(tfpl.DistributionLambda):
#     """An independent Weibull Keras layer."""

#     def __init__(
#         self,
#         event_shape=(),
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the `IndependentWeibull` layer.
#         Args:
#         event_shape: integer vector `Tensor` representing the shape of single
#             draw from this distribution.
#         convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#             instance and returns a `tf.Tensor`-like object.
#             Default value: `tfd.Distribution.mean`.
#         validate_args: Python `bool`, default `False`. When `True` distribution
#             parameters are checked for validity despite possibly degrading runtime
#             performance. When `False` invalid inputs may silently render incorrect
#             outputs.
#             Default value: `False`.
#         **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return IndependentWeibull.new(t, event_shape, validate_args)

#         super(IndependentWeibull, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_shape = event_shape
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_shape=(), validate_args=False, name=None):
#         """Create the distribution instance from a `params` vector."""
#         with tf.name_scope(name or "IndependentWeibull"):
#             params = tf.convert_to_tensor(params, name="params")
#             event_shape = dist_util.expand_to_vector(
#                 tf.convert_to_tensor(
#                     event_shape, name="event_shape", dtype_hint=tf.int32
#                 ),
#                 tensor_name="event_shape",
#             )
#             output_shape = tf.concat(
#                 [
#                     tf.shape(params)[:-1],
#                     event_shape,
#                 ],
#                 axis=0,
#             )
#             concentration, scale = tf.split(params, 2, axis=-1)
#             return independent_lib.Independent(
#                 tfd.Weibull(
#                     concentration=tf.math.softplus(
#                         tf.reshape(concentration, output_shape)
#                     )
#                     + 1.0,
#                     scale=tf.math.softplus(tf.reshape(scale, output_shape)),
#                     validate_args=validate_args,
#                 ),
#                 reinterpreted_batch_ndims=tf.size(event_shape),
#                 validate_args=validate_args,
#             )

#     @staticmethod
#     def params_size(event_shape=(), name=None):
#         """The number of `params` needed to create a single distribution."""
#         with tf.name_scope(name or "IndependentWeibull_params_size"):
#             event_shape = tf.convert_to_tensor(
#                 event_shape, name="event_shape", dtype_hint=tf.int32
#             )
#             return np.int32(2) * _event_size(
#                 event_shape, name=name or "IndependentWeibull_params_size"
#             )

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_shape": self._event_shape,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(IndependentWeibull, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(IndependentWeibull, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class MultivariateNormalDiag(tfpl.DistributionLambda):
#     """A `d`-variate normal Keras layer from `2* d` params,
#     with a diagonal scale matrix.
#     """

#     def __init__(
#         self,
#         event_size,
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         """Initialize the layer.
#         Args:
#             event_size: Scalar `int` representing the size of single draw from this
#               distribution.
#             convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
#               instance and returns a `tf.Tensor`-like object. For examples, see
#               `class` docstring.
#               Default value: `tfd.Distribution.sample`.
#             validate_args: Python `bool`, default `False`. When `True` distribution
#               parameters are checked for validity despite possibly degrading runtime
#               performance. When `False` invalid inputs may silently render incorrect
#               outputs.
#               Default value: `False`.
#             **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
#         """
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         def new_from_t(t):
#             return MultivariateNormalDiag.new(t, event_size, validate_args)

#         super(MultivariateNormalDiag, self).__init__(
#             new_from_t, convert_to_tensor_fn, **kwargs
#         )

#         self._event_size = event_size
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     @staticmethod
#     def new(params, event_size, validate_args=False, name=None):
#         """Create the distribution instance from a 'params' vector."""
#         with tf.name_scope(name or "MultivariateNormalDiag"):
#             params = tf.convert_to_tensor(params, name="params")
#             if event_size > 1:
#                 dist = tfd.MultivariateNormalDiag(
#                     loc=params[..., :event_size],
#                     scale_diag=1e-5 + tf.math.softplus(params[..., event_size:]),
#                     validate_args=validate_args,
#                 )
#             else:
#                 dist = tfd.Normal(
#                     loc=params[..., :event_size],
#                     scale=1e-5 + tf.math.softplus(params[..., event_size:]),
#                     validate_args=validate_args,
#                 )
#             return dist

#     @staticmethod
#     def params_size(event_size, name=None):
#         """The number of 'params' needed to create a single distribution."""
#         with tf.name_scope(name or "MultivariateNormalDiag_params_size"):
#             return 2 * event_size

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_size": self._event_size,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(MultivariateNormalDiag, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(MultivariateNormalDiag, self).output[0]


# @tf.keras.saving.register_keras_serializable()
# class MultivariateNormalTriL(tfpl.MultivariateNormalTriL):
#     def __init__(
#         self,
#         event_size,
#         convert_to_tensor_fn=tfd.Distribution.mean,
#         validate_args=False,
#         **kwargs
#     ):
#         convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

#         # If there is a 'make_distribution_fn' keyword argument (e.g., because we
#         # are being called from a `from_config` method), remove it.  We pass the
#         # distribution function to `DistributionLambda.__init__` below as the first
#         # positional argument.
#         kwargs.pop("make_distribution_fn", None)

#         super().__init__(event_size, convert_to_tensor_fn, validate_args, **kwargs)
#         self._event_size = event_size
#         self._convert_to_tensor_fn = convert_to_tensor_fn
#         self._validate_args = validate_args

#     def get_config(self):
#         """Returns the config of this layer.
#         NOTE: At the moment, this configuration can only be serialized if the
#         Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
#         implements `get_config`) or one of the standard values:
#         - `Distribution.sample` (or `"sample"`)
#         - `Distribution.mean` (or `"mean"`)
#         - `Distribution.mode` (or `"mode"`)
#         - `Distribution.stddev` (or `"stddev"`)
#         - `Distribution.variance` (or `"variance"`)
#         """
#         config = {
#             "event_size": self._event_size,
#             "convert_to_tensor_fn": _serialize(self._convert_to_tensor_fn),
#             "validate_args": self._validate_args,
#         }
#         base_config = super(MultivariateNormalTriL, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @property
#     def output(self):
#         """This allows the use of this layer with the shap package."""
#         return super(MultivariateNormalTriL, self).output[0]
