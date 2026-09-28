"""Loss functions for probabilistic (and deterministic) models.

Most losses wrap scoring rules from the `scoringrules` package, either in their
closed form for a given parametric distribution (`DistributionLossWrapper`), or
estimated from samples of the predicted distribution (`SampleLossWrapper`).

All losses in this module:

- accept as `y_pred` a `WrappingTorchDist` (or a `torch.distributions.Distribution`),
  as returned by the models in `mlpp_lib.models`, or a tensor;
- ignore non-finite targets (e.g. missing observations) in `y_true`;
- support `sample_weight` and the usual keras reductions.
"""

import importlib
import math
import re
import typing as tp
import warnings
from contextlib import contextmanager
from typing import Literal, Optional, Union

import keras
import scoringrules as sr
import torch
from keras import ops
from keras.saving import register_keras_serializable
from torch.distributions import Distribution

from mlpp_lib.custom_distributions import (
    _finite_times,
    _standardize,
    _std_cdf,
    _std_log_mass,
    _std_logpdf,
    _std_pdf,
)
from mlpp_lib.exceptions import MissingReparameterizationError
from mlpp_lib.probabilistic_layers import WrappingTorchDist

Prediction = Union[WrappingTorchDist, Distribution, torch.Tensor]


def _is_distribution(x) -> bool:
    return isinstance(x, (Distribution, WrappingTorchDist))


def _wrap(dist: Union[WrappingTorchDist, Distribution]) -> WrappingTorchDist:
    return dist if isinstance(dist, WrappingTorchDist) else WrappingTorchDist(dist)


def _fn_to_path(fn: tp.Callable) -> str:
    """Returns an importable dotted path for a function, preferring the public
    `scoringrules.<name>` path for scoringrules functions."""
    if getattr(sr, fn.__name__, None) is fn:
        return f"scoringrules.{fn.__name__}"
    return f"{fn.__module__}.{fn.__qualname__}"


def _path_to_fn(path: str) -> tp.Callable:
    module_name, fn_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), fn_name)


@contextmanager
def _scoringrules_backend(backend: str):
    """Temporarily set the active scoringrules backend. Some scoringrules functions
    use the active backend internally instead of the `backend` argument."""
    previous = getattr(sr.backends, "_active", None)
    sr.backends.set_active(backend)
    try:
        yield
    finally:
        if previous is not None:
            sr.backends.set_active(previous)


def _get_samples(y_pred: Prediction, num_samples: int) -> torch.Tensor:
    """Returns samples of shape [batch, samples, ...] from a predicted distribution.
    Tensors are interpreted as ensembles of shape [batch, members, ...]."""
    if not _is_distribution(y_pred):
        return y_pred
    y_pred = _wrap(y_pred)
    if y_pred.has_rsample:
        return y_pred.rsample(num_samples, pattern="bsd")
    if torch.is_grad_enabled():
        raise MissingReparameterizationError(
            f"Gradient-based optimization will not work, {y_pred.name} "
            "does not implement rsample()."
        )
    return y_pred.sample(num_samples, pattern="bsd")


class DistributionLoss(keras.Loss):
    """Loss base class allowing for non-tensor (distributions) predictions.

    We override the `__call__` method in order to avoid calling `convert_to_tensor`
    on `Distribution` objects, which would raise an error. Non-finite values in
    `y_true` (e.g. missing observations) are ignored in the computation of the loss.

    Subclasses implement `call(y_true, y_pred)`, which returns the loss either for
    each element of `y_true` (shape [batch, dim]) or for each sample (shape [batch]).
    """

    def __call__(self, y_true, y_pred, sample_weight=None):
        with keras.name_scope(self.name):
            if not _is_distribution(y_pred):
                y_pred = ops.convert_to_tensor(y_pred, dtype=self.dtype)
            y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
            if y_true.device.type == "meta":
                # symbolic tracing by keras (e.g. to build the compiled loss):
                # only the output spec (a scalar, unless reduction="none") matters
                shape = y_true.shape if self.reduction in (None, "none") else ()
                return torch.zeros(shape, dtype=y_true.dtype, device="meta")

            # mask non-finite targets, replacing them with a value in the
            # support of the distribution to avoid (masked) infinite losses,
            # which would still produce NaN gradients
            # (no data-dependent control flow, to support symbolic tracing)
            mask = torch.isfinite(y_true)
            self._mask = mask
            if _is_distribution(y_pred):
                fill = torch.nan_to_num(_wrap(y_pred).mean.detach()).expand_as(y_true)
            else:
                fill = torch.zeros_like(y_true)
            y_true = torch.where(mask, y_true, fill)

            # scoringrules creates some tensors without specifying the device,
            # so we set the default device to the one of the data
            with _scoringrules_backend(keras.backend.backend()), torch.device(
                y_true.device
            ):
                losses = self.call(y_true, y_pred)
            return self._reduce(losses, mask, sample_weight)

    def _reduce(self, losses, mask, sample_weight):
        # align the mask with the losses, which can be per-element or per-sample
        if mask.ndim > losses.ndim:
            mask = mask.reshape(*mask.shape[: losses.ndim], -1).all(-1)
        mask = mask.expand_as(losses)
        weights = mask.to(losses.dtype)
        if sample_weight is not None:
            sample_weight = ops.convert_to_tensor(sample_weight, dtype=losses.dtype)
            while sample_weight.ndim > losses.ndim and sample_weight.shape[-1] == 1:
                sample_weight = sample_weight.squeeze(-1)
            while sample_weight.ndim < losses.ndim:
                sample_weight = sample_weight[..., None]
            weights = weights * sample_weight

        # use `where` to prevent NaNs/infs of masked entries from propagating
        weighted = torch.where(mask, losses * weights, torch.zeros_like(losses))
        if self.reduction in (None, "none"):
            return weighted
        total = weighted.sum()
        if self.reduction == "sum":
            return total
        if self.reduction == "sum_over_batch_size":
            return total / mask.sum().clamp_min(1)
        if self.reduction == "mean":
            return total / weights.sum().clamp_min(torch.finfo(losses.dtype).tiny)
        raise ValueError(f"Unsupported reduction '{self.reduction}'.")


def _closed_form_params(dist: Distribution) -> tuple[tuple, dict]:
    name = type(dist).__name__
    try:
        extract = SR_PARAMS[name]
    except KeyError:
        raise ValueError(
            f"No closed-form scoringrules parametrization is available for the "
            f"{name} distribution. Use a sample-based loss instead, such as "
            "`CRPSEnsemble` (univariate) or `EnergyScore` (multivariate)."
        ) from None
    return extract(dist)


def crps_truncated_normal(obs, loc, scale, low, high, backend=None):
    """Closed-form CRPS of a normal distribution truncated to [low, high].

    Numerically stable version of the expression in Jordan et al. (2019),
    "Evaluating Probabilistic Forecasts with scoringRules", also far in the tails
    and for infinite bounds. Only supports torch tensors.
    """
    low, high = torch.as_tensor(low).to(loc), torch.as_tensor(high).to(loc)
    alpha, beta = _standardize(low, loc, scale), _standardize(high, loc, scale)
    y = (obs - loc) / scale
    z = torch.minimum(torch.maximum(y, alpha), beta)
    log_mass = _std_log_mass(alpha, beta)
    cdf_z = torch.exp(_std_log_mass(alpha, z) - log_mass)
    sqrt2 = math.sqrt(2.0)
    crps = (
        torch.abs(y - z)
        + z * (2 * cdf_z - 1)
        + 2 * torch.exp(_std_logpdf(z) - log_mass)
        - torch.exp(_std_log_mass(sqrt2 * alpha, sqrt2 * beta) - 2 * log_mass)
        / math.sqrt(math.pi)
    )
    return scale * crps


def crps_censored_normal(obs, loc, scale, low, high, backend=None):
    """Closed-form CRPS of a normal distribution censored to [low, high].

    Numerically stable version of the expression in Jordan et al. (2019),
    "Evaluating Probabilistic Forecasts with scoringRules", also far in the tails
    and for infinite bounds. Only supports torch tensors.
    """
    low, high = torch.as_tensor(low).to(loc), torch.as_tensor(high).to(loc)
    alpha, beta = _standardize(low, loc, scale), _standardize(high, loc, scale)
    y = (obs - loc) / scale
    z = torch.minimum(torch.maximum(y, alpha), beta)
    mass_low, mass_high = _std_cdf(alpha), _std_cdf(-beta)
    sqrt2 = math.sqrt(2.0)
    crps = (
        torch.abs(y - z)
        + _finite_times(beta, mass_high**2)
        - _finite_times(alpha, mass_low**2)
        + z * (2 * _std_cdf(z) - 1)
        + 2 * _std_pdf(z)
        - 2 * _std_pdf(beta) * mass_high
        - 2 * _std_pdf(alpha) * mass_low
        - (_std_cdf(sqrt2 * beta) - _std_cdf(sqrt2 * alpha)) / math.sqrt(math.pi)
    )
    return scale * crps


def _mixture_normal_params(dist):
    components = dist.component_distribution
    if not isinstance(components, torch.distributions.Normal):
        raise ValueError("Only mixtures of normal distributions are supported.")
    return (
        (components.loc, components.scale, dist.mixture_distribution.probs),
        {"m_axis": -1},
    )


#: Mapping between the (base) distribution class name and a function returning
#: the positional and keyword arguments expected by the scoringrules functions,
#: e.g. `scoringrules.crps_normal(obs, *args, **kwargs)`.
SR_PARAMS: dict[str, tp.Callable[[Distribution], tuple[tuple, dict]]] = {
    "Normal": lambda d: ((d.loc, d.scale), {}),
    "LogisticDistribution": lambda d: ((d.loc, d.scale), {}),
    "LogNormal": lambda d: ((d.loc, d.scale), {}),
    "TruncatedNormalDistribution": lambda d: ((d.loc, d.scale, d.low, d.high), {}),
    "CensoredNormalDistribution": lambda d: ((d.loc, d.scale, d.low, d.high), {}),
    "Gamma": lambda d: ((d.concentration, d.rate), {}),
    "Beta": lambda d: ((d.concentration1, d.concentration0), {}),
    "Exponential": lambda d: ((d.rate,), {}),
    "Poisson": lambda d: ((d.rate,), {}),
    "Bernoulli": lambda d: ((d.probs,), {}),
    "MixtureSameFamily": _mixture_normal_params,
}


@register_keras_serializable(package="mlpp_lib.losses")
class DistributionLossWrapper(DistributionLoss):
    """
    Wraps a scoringrules score function with analytical formulation into a keras
    loss function, such that it can be used with y_true being a tensor and y_pred
    being a predicted distribution. This means that the loss value is computed
    directly from the parameters of the distribution rather than samples.

    Parameters
    ----------
    fn: callable or str
        The scoringrules function, e.g. `scoringrules.crps_normal`, or its path.
    **fn_kwargs:
        Extra keyword arguments passed to `fn`.
    """

    def __init__(
        self,
        fn: Union[tp.Callable, str],
        name: Optional[str] = None,
        reduction: str = "sum_over_batch_size",
        dtype=None,
        **fn_kwargs,
    ):
        self.fn = _path_to_fn(fn) if isinstance(fn, str) else fn
        self.fn_kwargs = fn_kwargs
        super().__init__(
            name=name or self.fn.__name__, reduction=reduction, dtype=dtype
        )

    def call(self, y_true, y_pred):
        if not _is_distribution(y_pred):
            raise TypeError(
                f"{type(self).__name__} requires a predicted distribution, "
                f"got {type(y_pred)}."
            )
        args, kwargs = _closed_form_params(_wrap(y_pred).base_distribution)
        return self.fn(
            y_true, *args, **kwargs, **self.fn_kwargs, backend=keras.backend.backend()
        )

    def get_config(self):
        return {
            **super().get_config(),
            "fn": _fn_to_path(self.fn),
            **self.fn_kwargs,
        }


@register_keras_serializable(package="mlpp_lib.losses")
class SampleLossWrapper(DistributionLoss):
    """
    Wraps a scoringrules ensemble-based score function into a keras loss function,
    such that it can be used with y_true being a tensor and y_pred being a predicted
    distribution. Internally, `num_samples` samples are drawn from the predicted
    distribution and the loss value is estimated with a Monte Carlo approach.
    For gradient-based optimization, the distribution must support reparametrized
    sampling (`rsample`).

    If `y_pred` is a tensor, it is interpreted as an ensemble of shape
    [batch, members, ...].

    Parameters
    ----------
    fn: callable or str
        The scoringrules function, e.g. `scoringrules.crps_ensemble`, or its path.
    num_samples: int
        The number of samples drawn from the predicted distribution.
    estimator: str
        The estimator used by scoringrules (e.g. "pwm", "fair", "qd", "nrg").
        Prefer estimators whose memory grows linearly with `num_samples`, such as
        "pwm" (unbiased) and "qd" (biased) for the CRPS: "fair" and "nrg" give the
        same values but compare all pairs of samples.
        If None, the scoringrules default is used.
    **fn_kwargs:
        Extra keyword arguments passed to `fn`.
    """

    def __init__(
        self,
        fn: Union[tp.Callable, str],
        num_samples: int = 21,
        estimator: Optional[str] = "pwm",
        name: Optional[str] = None,
        reduction: str = "sum_over_batch_size",
        dtype=None,
        **fn_kwargs,
    ):
        self.fn = _path_to_fn(fn) if isinstance(fn, str) else fn
        self.num_samples = int(num_samples)
        if self.num_samples < 2:
            raise ValueError("num_samples must be > 1")
        self.estimator = estimator
        self.fn_kwargs = fn_kwargs
        super().__init__(
            name=name or self.fn.__name__, reduction=reduction, dtype=dtype
        )

    def call(self, y_true, y_pred):
        kwargs = dict(self.fn_kwargs)
        if self.estimator is not None:
            kwargs["estimator"] = self.estimator
        return self.fn(
            y_true,
            _get_samples(y_pred, self.num_samples),
            m_axis=1,
            **kwargs,
            backend=keras.backend.backend(),
        )

    def get_config(self):
        return {
            **super().get_config(),
            "fn": _fn_to_path(self.fn),
            "num_samples": self.num_samples,
            "estimator": self.estimator,
            **self.fn_kwargs,
        }


class _ClosedFormCRPS(DistributionLossWrapper):
    """Base class for the closed-form CRPS of a given distribution.

    Note: there is no named closed-form CRPS for the Gamma and Beta distributions,
    since the torch backend of scoringrules does not support their gradients
    (Gamma) or evaluation (Beta). Use `CRPSEnsemble` to train them instead.
    """

    score_fn: tp.Callable

    def __init__(self, **kwargs):
        kwargs.pop("fn", None)
        kwargs.setdefault("name", _snake_case(type(self).__name__))
        super().__init__(fn=type(self).score_fn, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop("fn")
        return config


def _snake_case(name: str) -> str:
    """CRPSTruncatedNormal -> crps_truncated_normal"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSNormal(_ClosedFormCRPS):
    """Closed-form CRPS of a `Normal` distribution."""

    score_fn = staticmethod(sr.crps_normal)


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSLogistic(_ClosedFormCRPS):
    """Closed-form CRPS of a `Logistic` distribution."""

    score_fn = staticmethod(sr.crps_logistic)


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSLogNormal(_ClosedFormCRPS):
    """Closed-form CRPS of a `LogNormal` distribution."""

    score_fn = staticmethod(sr.crps_lognormal)


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSTruncatedNormal(_ClosedFormCRPS):
    """Closed-form CRPS of a `TruncatedNormal` distribution."""

    score_fn = staticmethod(crps_truncated_normal)


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSCensoredNormal(_ClosedFormCRPS):
    """Closed-form CRPS of a `CensoredNormal` distribution."""

    score_fn = staticmethod(crps_censored_normal)


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSExponential(_ClosedFormCRPS):
    """Closed-form CRPS of an `Exponential` distribution."""

    score_fn = staticmethod(sr.crps_exponential)


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSPoisson(_ClosedFormCRPS):
    """Closed-form CRPS of a `Poisson` distribution."""

    score_fn = staticmethod(sr.crps_poisson)


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSMixtureNormal(_ClosedFormCRPS):
    """Closed-form CRPS of a `MixtureNormal` distribution."""

    score_fn = staticmethod(sr.crps_mixnorm)


class _SampleScore(SampleLossWrapper):
    """Base class for the named sample-based scores."""

    score_fn: tp.Callable

    def __init__(self, **kwargs):
        kwargs.pop("fn", None)
        kwargs.setdefault("name", _snake_case(type(self).__name__))
        super().__init__(fn=type(self).score_fn, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop("fn")
        return config


@register_keras_serializable(package="mlpp_lib.losses")
class CRPSEnsemble(_SampleScore):
    """CRPS estimated from `num_samples` samples of the predicted distribution."""

    score_fn = staticmethod(sr.crps_ensemble)


@register_keras_serializable(package="mlpp_lib.losses")
class TWCRPSEnsemble(_SampleScore):
    """Threshold-weighted CRPS, with weight function 1{a < x < b}, estimated from
    `num_samples` samples of the predicted distribution."""

    score_fn = staticmethod(sr.twcrps_ensemble)

    def __init__(self, a: float = float("-inf"), b: float = float("inf"), **kwargs):
        super().__init__(a=float(a), b=float(b), **kwargs)


@register_keras_serializable(package="mlpp_lib.losses")
class WeightedCRPSEnergy(TWCRPSEnsemble):
    """Threshold-weighted CRPS with weight function 1{x > threshold}, as in
    mlpp-lib < 1.0. Equivalent to `TWCRPSEnsemble(a=threshold)`.

    Parameters
    ----------
    threshold: float
        The threshold.
    num_samples: int
        The number of samples drawn from the predicted distribution.
    n_samples: int
        Deprecated alias of `num_samples`.
    correct_crps: bool
        Deprecated, use `estimator` instead. `True` (the default) corresponds to the
        unbiased "pwm" estimator, `False` to the biased "qd" estimator.
    """

    def __init__(
        self,
        threshold: float = float("-inf"),
        num_samples: Optional[int] = None,
        n_samples: Optional[int] = None,
        correct_crps: Optional[bool] = None,
        **kwargs,
    ):
        kwargs.pop("a", None)
        kwargs.pop("b", None)
        num_samples = _resolve_num_samples(num_samples, n_samples, default=1000)
        if correct_crps is not None:
            warnings.warn(
                "`correct_crps` is deprecated, use `estimator` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("estimator", "pwm" if correct_crps else "qd")
        kwargs.setdefault("name", "weighted_crps_energy")
        super().__init__(a=threshold, num_samples=num_samples, **kwargs)
        self.threshold = float(threshold)

    def get_config(self):
        config = super().get_config()
        config.pop("a")
        config.pop("b")
        return {**config, "threshold": self.threshold}


@register_keras_serializable(package="mlpp_lib.losses")
class EnergyScore(_SampleScore):
    """Energy score of a multivariate predicted distribution, estimated from
    `num_samples` samples.

    Parameters
    ----------
    num_samples: int
        The number of samples drawn from the predicted distribution.
    n_samples: int
        Deprecated alias of `num_samples`.
    estimator: str
        The scoringrules estimator. The default "akr_circperm" compares each sample
        with a shifted copy of the ensemble, so its memory grows linearly with
        `num_samples`, and it is unbiased for samples drawn from the predicted
        distribution. "nrg" and "fair" compare all pairs of samples, which needs
        memory proportional to batch_size * num_samples**2.
    """

    # `energy_score` was renamed to `es_ensemble` in scoringrules 0.10
    score_fn = staticmethod(getattr(sr, "es_ensemble", None) or sr.energy_score)

    def __init__(
        self,
        num_samples: Optional[int] = None,
        n_samples: Optional[int] = None,
        estimator: str = "akr_circperm",
        **kwargs,
    ):
        num_samples = _resolve_num_samples(num_samples, n_samples, default=100)
        super().__init__(
            num_samples=num_samples, estimator=estimator, v_axis=-1, **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.pop("v_axis")
        return config


def _resolve_num_samples(num_samples, n_samples, default):
    if n_samples is not None:
        warnings.warn(
            "`n_samples` is deprecated, use `num_samples` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if num_samples is None:
            num_samples = n_samples
    return default if num_samples is None else num_samples


@register_keras_serializable(package="mlpp_lib.losses")
class NegativeLogLikelihood(DistributionLoss):
    """Negative log-likelihood (log score) of the predicted distribution."""

    def __init__(self, name="negative_log_likelihood", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        if not _is_distribution(y_pred):
            raise TypeError(
                f"{type(self).__name__} requires a predicted distribution, "
                f"got {type(y_pred)}."
            )
        # use the base distribution to get element-wise values for
        # (independent) multi-output distributions
        return -_wrap(y_pred).base_distribution.log_prob(y_true)


@register_keras_serializable(package="mlpp_lib.losses")
class MultivariateLoss(DistributionLoss):
    """
    Compute losses for multivariate data.

    Facilitates computing losses for multivariate targets
    that may have different units. Allows rescaling the inputs
    and applying weights to each target variables.

    Parameters
    ----------
    metric: {"mse", "mae", "crps_energy"}
        The function used to compute the loss. For predicted distributions,
        "mse" and "mae" are computed on the mean, and "crps_energy" is estimated
        from `num_samples` samples.
    scaling: {"minmax", "standard"}
        (Optional) A scaling to apply to the data, in order to address the differences
        in magnitude due to different units. The statistics are computed on the
        targets of each batch. Default is `None`.
    weights: array-like
        (Optional) Weights assigned to each variable in the computation of the loss.
        Default is `None`.
    num_samples: int
        Number of samples used for "crps_energy".
    **kwargs:
        (Optional) Additional keyword arguments to be passed to the parent `Loss` class.
    """

    avail_metrics = ("mse", "mae", "crps_energy")
    avail_scaling = ("minmax", "standard", None)

    def __init__(
        self,
        metric: Literal["mse", "mae", "crps_energy"],
        scaling: Optional[Literal["minmax", "standard"]] = None,
        weights: Optional[list] = None,
        num_samples: int = 100,
        name: str = "multivariate_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if metric not in self.avail_metrics:
            raise NotImplementedError(
                f"`metric` argument must be one of {self.avail_metrics}"
            )
        if scaling not in self.avail_scaling:
            raise NotImplementedError(
                f"`scaling` argument must be one of {self.avail_scaling}"
            )
        self.metric = metric
        self.scaling = scaling
        self.weights = list(weights) if weights is not None else None
        self.num_samples = num_samples

    def get_config(self):
        return {
            **super().get_config(),
            "metric": self.metric,
            "scaling": self.scaling,
            "weights": self.weights,
            "num_samples": self.num_samples,
        }

    def _scaling_stats(self, y_true):
        mask = self._mask
        count = mask.sum(0).clamp_min(1)
        if self.scaling == "standard":
            mean = torch.where(mask, y_true, 0).sum(0) / count
            var = torch.where(mask, (y_true - mean) ** 2, 0).sum(0) / count
            return mean, var.sqrt().clamp_min(torch.finfo(y_true.dtype).eps)
        # minmax
        inf = torch.tensor(float("inf"), dtype=y_true.dtype, device=y_true.device)
        y_min = torch.where(mask, y_true, inf).amin(0)
        y_max = torch.where(mask, y_true, -inf).amax(0)
        return y_min, (y_max - y_min).clamp_min(torch.finfo(y_true.dtype).eps)

    def call(self, y_true, y_pred):
        if self.weights is not None and len(self.weights) != y_true.shape[-1]:
            raise ValueError(
                "Number weights must match the number of target variables."
            )

        if self.metric == "crps_energy":
            y_pred = _get_samples(y_pred, self.num_samples)  # [B, S, D]
        elif _is_distribution(y_pred):
            y_pred = _wrap(y_pred).mean

        if self.scaling is not None:
            shift, scale = self._scaling_stats(y_true)
            y_true = (y_true - shift) / scale
            y_pred = (y_pred - shift) / scale

        if self.metric == "mse":
            loss = (y_true - y_pred) ** 2
        elif self.metric == "mae":
            loss = torch.abs(y_true - y_pred)
        else:
            loss = sr.crps_ensemble(
                y_true, y_pred, m_axis=1, estimator="pwm", backend="torch"
            )

        if self.weights is not None:
            loss = loss * torch.as_tensor(
                self.weights, dtype=loss.dtype, device=loss.device
            )
        return loss


@register_keras_serializable(package="mlpp_lib.losses")
class CombinedLoss(DistributionLoss):
    """Weighted sum of losses.

    Parameters
    ----------
    losses: list[dict]
        List of loss configurations (see `mlpp_lib.utils.get_loss`), each with an
        optional `weight` key (default 1.0), e.g.
        `[{"CRPSNormal": {}, "weight": 0.5}, {"MultivariateLoss": {"metric": "mse"}}]`.
    """

    def __init__(self, losses: list, name: str = "combined_loss", **kwargs):
        # local import to avoid circular dependency with mlpp_lib.utils
        from mlpp_lib.utils import get_loss

        super().__init__(name=name, **kwargs)
        self.losses_config = losses
        self.weights = []
        self.losses = []
        for loss_config in losses:
            if isinstance(loss_config, dict):
                loss_config = dict(loss_config)
                self.weights.append(float(loss_config.pop("weight", 1.0)))
            else:
                self.weights.append(1.0)
            self.losses.append(get_loss(loss_config))

    def __call__(self, y_true, y_pred, sample_weight=None):
        return sum(
            weight * loss(y_true, y_pred, sample_weight=sample_weight)
            for loss, weight in zip(self.losses, self.weights)
        )

    def call(self, y_true, y_pred):
        raise NotImplementedError("CombinedLoss is computed in __call__.")

    def get_config(self):
        return {**super().get_config(), "losses": self.losses_config}
