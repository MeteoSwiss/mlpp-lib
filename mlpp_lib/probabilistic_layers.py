"""Probabilistic output layers based on `torch.distributions`.

A `DistributionLayer` maps its inputs (e.g. the output of an encoder) to the
parameters of a parametric distribution, and returns the distribution itself
(wrapped in a `WrappingTorchDist`), samples from it, or its expected value.

The parametric distributions are defined as subclasses of
`BaseParametricDistribution`, which are responsible for mapping the raw
(unconstrained) parameters to a valid `torch.distributions.Distribution`
(e.g. ensuring positive scales). They are referred to by name, see
`DISTRIBUTIONS` for a list of the available ones.
"""

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import keras
import numpy as np
import torch
import torch.nn.functional as F
from keras import initializers
from torch.distributions import Distribution, Independent
from torch.distributions.utils import vec_to_tril_matrix

from mlpp_lib.custom_distributions import (
    CensoredNormalDistribution,
    LogisticDistribution,
    TruncatedNormalDistribution,
)
from mlpp_lib.exceptions import MissingReparameterizationError

# Floor added to positive parameters (scales, rates, concentrations) for
# numerical stability, as in mlpp-lib < 1.0.
POSITIVE_EPS = 1e-3


def _positive(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x) + POSITIVE_EPS


def _as_tensor_like(value, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


class WrappingTorchDist:
    """
    Wraps a torch.distributions.Distribution instance.
    Unifies sample(torch.Size) and sample_n(int) in a single function and
    allows to specify a pattern for the samples between [Samples, Batch, Dim]
    ("sbd", the torch default) and [Batch, Samples, Dim] ("bsd").

    Any other attribute (e.g. `log_prob`, `cdf`, `variance`, `batch_shape`)
    is forwarded to the wrapped distribution.
    """

    def __init__(self, distribution: Distribution):
        self._distribution = distribution

    def __getattr__(self, name):
        # only called when the attribute is not found on the wrapper itself
        if name.startswith("__") or name == "_distribution":
            raise AttributeError(name)
        return getattr(self._distribution, name)

    @staticmethod
    def _sample_shape(n: Union[int, tuple]) -> torch.Size:
        return torch.Size((n,)) if isinstance(n, int) else torch.Size(n)

    def _get_samples(self, sampling_fn, n, pattern: Literal["sbd", "bsd"] = "sbd"):
        sample_shape = self._sample_shape(n)
        samples = sampling_fn(sample_shape)
        if pattern == "sbd":
            return samples
        if pattern == "bsd":
            k = len(sample_shape)
            return samples.movedim(tuple(range(k)), tuple(range(1, k + 1)))
        raise ValueError(f"Unknown sampling pattern '{pattern}'. Use 'sbd' or 'bsd'.")

    def sample(self, n: Union[int, tuple], pattern: Literal["sbd", "bsd"] = "sbd"):
        return self._get_samples(self._distribution.sample, n=n, pattern=pattern)

    def rsample(self, n: Union[int, tuple], pattern: Literal["sbd", "bsd"] = "sbd"):
        if not self.has_rsample:
            raise MissingReparameterizationError(
                f"{self.name} does not implement rsample."
            )
        return self._get_samples(self._distribution.rsample, n=n, pattern=pattern)

    def __repr__(self):
        return f"WrappingTorchDist({self._distribution!r})"

    @property
    def base_distribution(self) -> Distribution:
        """The underlying distribution, unwrapping `Independent` if needed."""
        dist = self._distribution
        while isinstance(dist, Independent):
            dist = dist.base_dist
        return dist

    @property
    def name(self) -> str:
        return type(self.base_distribution).__name__

    @property
    def has_rsample(self) -> bool:
        return self._distribution.has_rsample

    @property
    def mean(self):
        return self._distribution.mean


class BaseParametricDistribution(ABC):
    """Base class for parametric distributions.

    A parametric distribution maps a tensor of raw, unconstrained parameters of
    shape [..., num_parameters] to a `torch.distributions.Distribution` with
    `batch_shape=[...]` and `event_shape=[event_size]`.

    By default, the raw parameters are laid out as `num_params_per_event` blocks
    of `event_size` values (e.g. `[loc_1, ..., loc_n, scale_1, ..., scale_n]`),
    and every event dimension is modelled independently.
    """

    #: name used to refer to the distribution in configurations
    name: str
    #: the `torch.distributions.Distribution` class of the (base) distribution
    distribution_cls: type
    #: number of parameters for each (independent) event dimension
    num_params_per_event: int

    def __init__(self, event_size: int = 1):
        self.event_size = int(event_size)

    @property
    def num_parameters(self) -> int:
        """The number of raw parameters that describe the distribution."""
        return self.num_params_per_event * self.event_size

    @property
    def has_rsample(self) -> bool:
        return self.distribution_cls.has_rsample

    def get_config(self) -> dict[str, Any]:
        return {"event_size": self.event_size}

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(**config)

    def _split_params(self, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """[..., P * E] -> P tensors of shape [..., E]"""
        params = params.reshape(
            *params.shape[:-1], self.num_params_per_event, self.event_size
        )
        return params.unbind(-2)

    @abstractmethod
    def base_distribution(self, params: torch.Tensor) -> Distribution:
        """Given the raw parameters, returns the distribution with
        `batch_shape=[..., event_size]`, ensuring the parameters' constraints."""

    def process_params(self, params: torch.Tensor) -> Distribution:
        """Given the raw parameters predicted by a previous layer,
        returns the parametric distribution."""
        return Independent(self.base_distribution(params), 1, validate_args=False)

    def __call__(self, params: torch.Tensor) -> WrappingTorchDist:
        return WrappingTorchDist(self.process_params(params))

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"{type(self).__name__}({args})"


class ParametricNormal(BaseParametricDistribution):
    """Normal distribution with parameters (loc, scale)."""

    name = "Normal"
    distribution_cls = torch.distributions.Normal
    num_params_per_event = 2

    def base_distribution(self, params):
        loc, scale = self._split_params(params)
        return self.distribution_cls(loc, _positive(scale), validate_args=False)


class ParametricLogistic(BaseParametricDistribution):
    """Logistic distribution with parameters (loc, scale)."""

    name = "Logistic"
    distribution_cls = LogisticDistribution
    num_params_per_event = 2

    def base_distribution(self, params):
        loc, scale = self._split_params(params)
        return self.distribution_cls(loc, _positive(scale), validate_args=False)


class ParametricLogNormal(BaseParametricDistribution):
    """LogNormal distribution Y = exp(X), X ~ Normal(loc, scale)."""

    name = "LogNormal"
    distribution_cls = torch.distributions.LogNormal
    num_params_per_event = 2

    def base_distribution(self, params):
        loc, scale = self._split_params(params)
        return self.distribution_cls(loc, _positive(scale), validate_args=False)


class ParametricTruncatedNormal(BaseParametricDistribution):
    """Normal distribution with parameters (loc, scale) truncated to [low, high]."""

    name = "TruncatedNormal"
    distribution_cls = TruncatedNormalDistribution
    num_params_per_event = 2

    def __init__(
        self, event_size: int = 1, low: float = 0.0, high: float = float("inf")
    ):
        super().__init__(event_size)
        if not low < high:
            raise ValueError(f"Expected low < high, got low={low}, high={high}.")
        self.low, self.high = float(low), float(high)

    def get_config(self):
        return {**super().get_config(), "low": self.low, "high": self.high}

    def base_distribution(self, params):
        loc, scale = self._split_params(params)
        return self.distribution_cls(
            loc,
            _positive(scale),
            low=_as_tensor_like(self.low, loc),
            high=_as_tensor_like(self.high, loc),
            validate_args=False,
        )


class ParametricCensoredNormal(ParametricTruncatedNormal):
    """Normal distribution with parameters (loc, scale) censored to [low, high]."""

    name = "CensoredNormal"
    distribution_cls = CensoredNormalDistribution


class ParametricGamma(BaseParametricDistribution):
    """Gamma distribution with parameters (concentration, rate)."""

    name = "Gamma"
    distribution_cls = torch.distributions.Gamma
    num_params_per_event = 2

    def base_distribution(self, params):
        concentration, rate = self._split_params(params)
        return self.distribution_cls(
            _positive(concentration), _positive(rate), validate_args=False
        )


class ParametricBeta(BaseParametricDistribution):
    """Beta distribution with parameters (concentration1, concentration0),
    or (alpha, beta)."""

    name = "Beta"
    distribution_cls = torch.distributions.Beta
    num_params_per_event = 2

    def base_distribution(self, params):
        c1, c0 = self._split_params(params)
        return self.distribution_cls(_positive(c1), _positive(c0), validate_args=False)


class ParametricWeibull(BaseParametricDistribution):
    """Weibull distribution with parameters (concentration, scale)."""

    name = "Weibull"
    distribution_cls = torch.distributions.Weibull
    num_params_per_event = 2

    def base_distribution(self, params):
        concentration, scale = self._split_params(params)
        return self.distribution_cls(
            scale=_positive(scale),
            concentration=_positive(concentration),
            validate_args=False,
        )


class ParametricExponential(BaseParametricDistribution):
    """Exponential distribution with parameter (rate)."""

    name = "Exponential"
    distribution_cls = torch.distributions.Exponential
    num_params_per_event = 1

    def base_distribution(self, params):
        (rate,) = self._split_params(params)
        return self.distribution_cls(_positive(rate), validate_args=False)


class ParametricPoisson(BaseParametricDistribution):
    """Poisson distribution with parameter (rate). It does not support
    reparametrized sampling."""

    name = "Poisson"
    distribution_cls = torch.distributions.Poisson
    num_params_per_event = 1

    def base_distribution(self, params):
        (rate,) = self._split_params(params)
        return self.distribution_cls(_positive(rate), validate_args=False)


class ParametricBernoulli(BaseParametricDistribution):
    """Bernoulli distribution with parameter (logits). It does not support
    reparametrized sampling."""

    name = "Bernoulli"
    distribution_cls = torch.distributions.Bernoulli
    num_params_per_event = 1

    def base_distribution(self, params):
        (logits,) = self._split_params(params)
        return self.distribution_cls(logits=logits, validate_args=False)


class ParametricMixtureNormal(BaseParametricDistribution):
    """Mixture of `num_components` normal distributions, with parameters
    (loc_k, scale_k, logits_k) for each component k. It does not support
    reparametrized sampling."""

    name = "MixtureNormal"
    distribution_cls = torch.distributions.MixtureSameFamily

    def __init__(self, event_size: int = 1, num_components: int = 2):
        super().__init__(event_size)
        self.num_components = int(num_components)

    @property
    def num_params_per_event(self):
        return 3 * self.num_components

    def get_config(self):
        return {**super().get_config(), "num_components": self.num_components}

    def base_distribution(self, params):
        k, e = self.num_components, self.event_size
        # [..., 3 * K * E] -> [..., 3, E, K]
        params = params.reshape(*params.shape[:-1], 3, k, e).transpose(-1, -2)
        loc, scale, logits = params.unbind(-3)
        return self.distribution_cls(
            torch.distributions.Categorical(logits=logits, validate_args=False),
            torch.distributions.Normal(loc, _positive(scale), validate_args=False),
            validate_args=False,
        )


class ParametricMultivariateNormalTriL(BaseParametricDistribution):
    """Multivariate normal distribution ~N(loc, LL^T), parametrized by the mean vector
    and a lower triangular matrix L (the Cholesky factor of the covariance matrix).
    The diagonal of L is made positive with a softplus.
    """

    name = "MultivariateNormalTriL"
    distribution_cls = torch.distributions.MultivariateNormal

    @property
    def num_parameters(self):
        return self.event_size + self.event_size * (self.event_size + 1) // 2

    def process_params(self, params):
        loc = params[..., : self.event_size]
        scale_tril = vec_to_tril_matrix(params[..., self.event_size :])
        diag = torch.diagonal(scale_tril, dim1=-2, dim2=-1)
        scale_tril = (
            scale_tril - torch.diag_embed(diag) + torch.diag_embed(_positive(diag))
        )
        return self.distribution_cls(
            loc=loc, scale_tril=scale_tril, validate_args=False
        )

    def base_distribution(self, params):
        return self.process_params(params)


#: all available parametric distributions, by name
DISTRIBUTIONS: dict[str, type[BaseParametricDistribution]] = {
    cls.name: cls
    for cls in [
        ParametricNormal,
        ParametricLogistic,
        ParametricLogNormal,
        ParametricTruncatedNormal,
        ParametricCensoredNormal,
        ParametricGamma,
        ParametricBeta,
        ParametricWeibull,
        ParametricExponential,
        ParametricPoisson,
        ParametricBernoulli,
        ParametricMixtureNormal,
        ParametricMultivariateNormalTriL,
    ]
}

# Names used by mlpp-lib < 1.0 (tensorflow-probability layers), mapped to the
# new names and the options that reproduce their behaviour.
LEGACY_DISTRIBUTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "IndependentNormal": ("Normal", {}),
    "IndependentLogistic": ("Logistic", {}),
    "IndependentLogNormal": ("LogNormal", {}),
    "IndependentTruncatedNormal": ("TruncatedNormal", {"low": 0.0, "high": math.inf}),
    "IndependentDoublyCensoredNormal": ("CensoredNormal", {"low": 0.0, "high": 1.0}),
    "IndependentGamma": ("Gamma", {}),
    "IndependentBeta": ("Beta", {}),
    "IndependentWeibull": ("Weibull", {}),
    "IndependentPoisson": ("Poisson", {}),
    "IndependentBernoulli": ("Bernoulli", {}),
    "IndependentMixtureNormal": ("MixtureNormal", {"num_components": 2}),
    "MultivariateNormalDiag": ("Normal", {}),
}

# Options of the tensorflow-probability layers that no longer have an effect.
_LEGACY_OPTIONS = ("event_shape", "convert_to_tensor_fn", "validate_args")


def get_distribution(
    name: str, event_size: int = 1, **kwargs
) -> BaseParametricDistribution:
    """Get a parametric distribution by name.

    Names used by mlpp-lib < 1.0 (e.g. `IndependentNormal`) are still accepted,
    but deprecated.
    """
    if name in LEGACY_DISTRIBUTIONS:
        new_name, legacy_kwargs = LEGACY_DISTRIBUTIONS[name]
        warnings.warn(
            f"The distribution name '{name}' is deprecated, use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for option in _LEGACY_OPTIONS:
            if kwargs.pop(option, None) is not None:
                warnings.warn(
                    f"The option '{option}' of '{name}' is ignored.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        name, kwargs = new_name, {**legacy_kwargs, **kwargs}

    try:
        distribution_cls = DISTRIBUTIONS[name]
    except KeyError:
        raise KeyError(
            f"The distribution '{name}' is not available. "
            f"Choose one of {sorted(DISTRIBUTIONS)}."
        ) from None
    return distribution_cls(event_size=event_size, **kwargs)


def _serialize_distribution(distribution: BaseParametricDistribution) -> dict:
    return {"name": distribution.name, "config": distribution.get_config()}


def _deserialize_distribution(config: dict) -> BaseParametricDistribution:
    return DISTRIBUTIONS[config["name"]].from_config(config["config"])


@keras.saving.register_keras_serializable(package="mlpp_lib")
class DistributionLayer(keras.Layer):
    """
    Keras layer mapping its inputs to a parametric distribution.

    A linear layer maps the inputs into the number of (unconstrained) parameters of
    the underlying parametric distribution, which is then responsible to ensure the
    parameters' constraints, e.g. the positiveness of the scale.

    Parameters
    ----------
    distribution: str or BaseParametricDistribution
        The parametric distribution, or its name (see `DISTRIBUTIONS`).
    event_size: int
        The number of output dimensions (only used if `distribution` is a name).
    distribution_kwargs: dict
        Extra options for the distribution (only used if `distribution` is a name),
        e.g. `{"low": 0.0}` for a `TruncatedNormal`.
    num_samples: int
        The default number of samples, when the layer outputs samples.
    bias_init: str, keras initializer or array
        Initializer for the bias of the linear layer. If an array is passed,
        it is used for the first parameters (e.g. the `loc`) and padded with zeros.
    """

    def __init__(
        self,
        distribution: Union[str, BaseParametricDistribution],
        event_size: int = 1,
        distribution_kwargs: Optional[dict] = None,
        num_samples: int = 21,
        bias_init="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(distribution, str):
            distribution = get_distribution(
                distribution, event_size=event_size, **(distribution_kwargs or {})
            )
        elif not isinstance(distribution, BaseParametricDistribution):
            raise TypeError(
                "Expected a BaseParametricDistribution or the name of a distribution, "
                f"got {type(distribution)}."
            )
        self.distribution = distribution
        self.num_samples = num_samples

        if isinstance(bias_init, (list, tuple)):
            bias_init = np.asarray(bias_init, dtype=float)
        self.bias_init = bias_init

    @property
    def event_size(self) -> int:
        return self.distribution.event_size

    def _bias_initializer(self):
        if not isinstance(self.bias_init, np.ndarray):
            return self.bias_init
        bias = self.bias_init.ravel()
        num_params = self.distribution.num_parameters
        if bias.shape[0] > num_params:
            raise ValueError(
                f"bias_init has {bias.shape[0]} values, but the distribution "
                f"only has {num_params} parameters."
            )
        return initializers.Constant(np.pad(bias, (0, num_params - bias.shape[0])))

    def build(self, input_shape):
        self.parameters_encoder = keras.layers.Dense(
            self.distribution.num_parameters,
            name="parameters_encoder",
            bias_initializer=self._bias_initializer(),
        )
        self.parameters_encoder.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs,
        output_type: Literal["distribution", "samples", "expected"] = "distribution",
        num_samples: Optional[int] = None,
        pattern: Literal["sbd", "bsd"] = "sbd",
        training=None,
    ):
        dist = self.distribution(self.parameters_encoder(inputs))

        if output_type == "distribution":
            return dist
        if output_type == "expected":
            return dist.mean
        if output_type == "samples":
            num_samples = num_samples or self.num_samples
            if dist.has_rsample:
                return dist.rsample(num_samples, pattern=pattern)
            if training:
                raise MissingReparameterizationError(
                    "Gradient-based optimization will not work, as the underlying "
                    f"{dist.name} distribution does not have a reparametrized "
                    "sampling function."
                )
            return dist.sample(num_samples, pattern=pattern)
        raise ValueError(
            f"Unknown output_type '{output_type}'. "
            "Use 'distribution', 'samples' or 'expected'."
        )

    def compute_output_shape(self, input_shape):
        # shape of the expected value
        return (input_shape[0], self.event_size)

    def get_config(self):
        config = super().get_config()
        bias_init = self.bias_init
        if isinstance(bias_init, np.ndarray):
            bias_init = bias_init.tolist()
        elif not isinstance(bias_init, str):
            bias_init = keras.initializers.serialize(bias_init)
        config.update(
            {
                "distribution": _serialize_distribution(self.distribution),
                "num_samples": self.num_samples,
                "bias_init": bias_init,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["distribution"] = _deserialize_distribution(config["distribution"])
        if isinstance(config.get("bias_init"), dict):
            config["bias_init"] = keras.initializers.deserialize(config["bias_init"])
        return cls(**config)
