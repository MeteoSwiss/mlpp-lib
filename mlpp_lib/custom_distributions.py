"""Custom `torch.distributions` not (or not suitably) provided by torch itself.

All distributions support arbitrary batch shapes, infinite bounds and
reparametrized sampling, so they can be used both with closed-form and
sample-based (Monte Carlo) scoring rules.
"""

import math

import torch
from torch.distributions import (
    AffineTransform,
    Distribution,
    Normal,
    SigmoidTransform,
    TransformedDistribution,
    Uniform,
    constraints,
)
from torch.distributions.utils import broadcast_all

_SQRT_2PI = math.sqrt(2 * math.pi)


def _where_finite(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Replaces non-finite entries of `x` with `fill`. Used to keep both branches of
    `torch.where` finite, since NaN gradients leak through the unselected branch."""
    return torch.where(torch.isfinite(x), x, torch.full_like(x, fill))


def _standardize(bound: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
    """(bound - loc) / scale, returning the (constant) infinite bound unchanged so that
    no infinite values enter the autograd graph."""
    finite = torch.isfinite(bound)
    z = (_where_finite(bound) - loc) / scale
    return torch.where(finite, z, bound.expand_as(z))


def _std_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal pdf, returning 0 for infinite inputs."""
    xs = _where_finite(x)
    pdf = torch.exp(-0.5 * xs**2) / _SQRT_2PI
    return torch.where(torch.isfinite(x), pdf, torch.zeros_like(pdf))


def _std_cdf(x: torch.Tensor) -> torch.Tensor:
    return torch.special.ndtr(x)


def _x_pdf(x: torch.Tensor) -> torch.Tensor:
    """Computes x * pdf(x), with the (correct) limit 0 for infinite x."""
    xs = _where_finite(x)
    return torch.where(torch.isfinite(x), xs * _std_pdf(xs), torch.zeros_like(xs))


def _finite_times(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Computes x * w for bounds x that may be infinite, where w (a probability or a
    density at the bound) is 0 whenever x is infinite."""
    return _where_finite(x) * w


def _std_logpdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal log-pdf, returning -inf for infinite inputs."""
    xs = _where_finite(x)
    logpdf = -0.5 * xs**2 - math.log(_SQRT_2PI)
    return torch.where(torch.isfinite(x), logpdf, torch.full_like(logpdf, -torch.inf))


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x <= 0."""
    return torch.where(
        x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x))
    )


def _std_log_mass(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """log P(alpha < Z < beta) for a standard normal Z, stable also far in the tails."""
    upper_tail = alpha > 0
    log_hi = torch.where(
        upper_tail, torch.special.log_ndtr(-alpha), torch.special.log_ndtr(beta)
    )
    log_lo = torch.where(
        upper_tail, torch.special.log_ndtr(-beta), torch.special.log_ndtr(alpha)
    )
    return log_hi + _log1mexp(log_lo - log_hi)


def _std_mass(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """P(alpha < Z < beta) for a standard normal Z, computed in the tail where it is
    more accurate to avoid catastrophic cancellation."""
    upper_tail = alpha > 0
    return torch.where(
        upper_tail,
        _std_cdf(-alpha) - _std_cdf(-beta),
        _std_cdf(beta) - _std_cdf(alpha),
    )


class TruncatedNormalDistribution(Distribution):
    """Normal distribution with mean `loc` and standard deviation `scale`,
    truncated to the interval [low, high]. Bounds can be infinite.

    Source: The Truncated Normal Distribution, John Burkardt 2023
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: torch.Tensor | float = float("-inf"),
        high: torch.Tensor | float = float("inf"),
        validate_args=None,
    ):
        """
        Args:
            loc: the mean of the underlying normal. It is not the mean of the distribution.
            scale: the std of the underlying normal. It is not the std of the distribution.
            low: the lower bound.
            high: the upper bound.
        """
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        super().__init__(batch_shape=self.loc.shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        return type(self)(
            self.loc.expand(batch_shape),
            self.scale.expand(batch_shape),
            self.low.expand(batch_shape),
            self.high.expand(batch_shape),
            validate_args=False,
        )

    @property
    def _alpha(self):
        return _standardize(self.low, self.loc, self.scale)

    @property
    def _beta(self):
        return _standardize(self.high, self.loc, self.scale)

    @property
    def _log_mass(self):
        return _std_log_mass(self._alpha, self._beta)

    def _pdf_ratios(self):
        """pdf(alpha) / Z and pdf(beta) / Z, where Z = P(alpha < Z < beta),
        computed in log-space for numerical stability in the tails."""
        log_mass = self._log_mass
        r_a = torch.exp(_std_logpdf(self._alpha) - log_mass)
        r_b = torch.exp(_std_logpdf(self._beta) - log_mass)
        return r_a, r_b

    @property
    def mean(self) -> torch.Tensor:
        r_a, r_b = self._pdf_ratios()
        return self.loc + self.scale * (r_a - r_b)

    @property
    def variance(self) -> torch.Tensor:
        alpha, beta = self._alpha, self._beta
        r_a, r_b = self._pdf_ratios()
        return (
            self.scale**2
            * (
                1.0
                + _finite_times(alpha, r_a)
                - _finite_times(beta, r_b)
                - (r_a - r_b) ** 2
            )
        ).clamp_min(0.0)

    def moment(self, k: int) -> torch.Tensor:
        """Raw moment E[X^k], computed with the recursion from
        "A Recursive Formula for the Moments of a Truncated Univariate
        Normal Distribution" (Eric Orjebin)."""
        r_a, r_b = self._pdf_ratios()
        m_prev, m = torch.zeros_like(self.loc), torch.ones_like(self.loc)
        for i in range(1, k + 1):
            boundary = _finite_times(self.high ** (i - 1), r_b) - _finite_times(
                self.low ** (i - 1), r_a
            )
            m_prev, m = m, (
                (i - 1) * self.scale**2 * m_prev + self.loc * m - self.scale * boundary
            )
        return m

    def cdf(self, value):
        z = (value - self.loc) / self.scale
        z = torch.minimum(torch.maximum(z, self._alpha), self._beta)
        return torch.exp(_std_log_mass(self._alpha, z) - self._log_mass).clamp(0.0, 1.0)

    def icdf(self, p):
        alpha, beta = self._alpha, self._beta
        log_p_mass = torch.log(p.clamp_min(torch.finfo(p.dtype).tiny)) + self._log_mass
        # solve in log-space, in the tail where the standard normal cdf is most accurate
        log_sf_alpha = torch.special.log_ndtr(-alpha)
        log_q_upper = log_sf_alpha + _log1mexp(
            torch.minimum(log_p_mass - log_sf_alpha, torch.zeros_like(log_p_mass))
        )
        log_q_lower = torch.logaddexp(torch.special.log_ndtr(alpha), log_p_mass)
        upper_tail = alpha > 0
        q = torch.exp(torch.where(upper_tail, log_q_upper, log_q_lower))
        q = q.clamp(torch.finfo(q.dtype).tiny, 1.0 - torch.finfo(q.dtype).eps)
        z = torch.special.ndtri(q)
        z = torch.where(upper_tail, -z, z)
        z = torch.minimum(torch.maximum(z, alpha), beta)
        x = self.loc + self.scale * z
        # guard against round-off outside the support
        return torch.minimum(torch.maximum(x, self.low), self.high)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        log_prob = (
            -0.5 * z**2 - math.log(_SQRT_2PI) - torch.log(self.scale) - self._log_mass
        )
        inside = (value >= self.low) & (value <= self.high)
        return torch.where(inside, log_prob, torch.full_like(log_prob, -torch.inf))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(p)


class CensoredNormalDistribution(Distribution):
    r"""Normal distribution with mean `loc` and standard deviation `scale`,
    censored to the interval [low, high]: values of the underlying normal
    that lie outside the interval are assigned to `low` and `high`, respectively.
    Bounds can be infinite (e.g. `low=0, high=inf` for precipitation).

    .. math::
        Y =
            \begin{cases}
            a, & \text{if } X \leq a  \\
            X  & \text{if } a < X < b  \\
            b, & \text{if } X \geq b  \\
            \end{cases}
        \quad X \sim N(\mu, \sigma)

    The distribution is a mixture of a continuous and a discrete part (point masses
    at the bounds): `log_prob` returns the log-density inside the interval and the
    log-probability of the point masses at the bounds.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: torch.Tensor | float = 0.0,
        high: torch.Tensor | float = float("inf"),
        validate_args=None,
    ):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        self._normal = Normal(self.loc, self.scale, validate_args=False)
        super().__init__(batch_shape=self.loc.shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        return type(self)(
            self.loc.expand(batch_shape),
            self.scale.expand(batch_shape),
            self.low.expand(batch_shape),
            self.high.expand(batch_shape),
            validate_args=False,
        )

    def _std_bounds(self):
        alpha = _standardize(self.low, self.loc, self.scale)
        beta = _standardize(self.high, self.loc, self.scale)
        return alpha, beta

    @property
    def mean(self):
        alpha, beta = self._std_bounds()
        p_low, p_high = _std_cdf(alpha), _std_cdf(-beta)
        pdf_a, pdf_b = _std_pdf(alpha), _std_pdf(beta)
        # E[X 1{a<X<b}] for the underlying normal X
        inner = self.loc * _std_mass(alpha, beta) + self.scale * (pdf_a - pdf_b)
        return _finite_times(self.low, p_low) + _finite_times(self.high, p_high) + inner

    @property
    def variance(self):
        # Var(Y) = E(Y^2) - E(Y)^2, with
        # E(Y^2) = a^2 P(X<a) + b^2 P(X>b) + E(X^2 1{a<X<b})
        alpha, beta = self._std_bounds()
        p_low, p_high = _std_cdf(alpha), _std_cdf(-beta)
        pdf_a, pdf_b = _std_pdf(alpha), _std_pdf(beta)
        mass = _std_mass(alpha, beta)
        inner2 = (
            self.loc**2 * mass
            + 2 * self.loc * self.scale * (pdf_a - pdf_b)
            + self.scale**2 * (mass + _x_pdf(alpha) - _x_pdf(beta))
        )
        e_y2 = (
            _finite_times(self.low**2, p_low)
            + _finite_times(self.high**2, p_high)
            + inner2
        )
        return (e_y2 - self.mean**2).clamp_min(0.0)

    def cdf(self, value):
        cdf = self._normal.cdf(value)
        cdf = torch.where(value < self.low, torch.zeros_like(cdf), cdf)
        return torch.where(value >= self.high, torch.ones_like(cdf), cdf)

    def log_prob(self, value):
        alpha, beta = self._std_bounds()
        log_p_low = torch.special.log_ndtr(alpha)
        log_p_high = torch.special.log_ndtr(-beta)
        log_pdf = self._normal.log_prob(value)
        out = torch.where(value <= self.low, log_p_low, log_pdf)
        return torch.where(value >= self.high, log_p_high, out)

    def rsample(self, sample_shape=torch.Size()):
        # Pathwise gradient of the clipping: 1 inside the interval, 0 on the
        # point masses, which is the exact derivative of the censored sample.
        x = self._normal.rsample(sample_shape)
        return torch.minimum(torch.maximum(x, self.low), self.high)


class LogisticDistribution(TransformedDistribution):
    """Logistic distribution with location `loc` and scale `scale`
    (torch does not provide one natively)."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = torch.finfo(self.loc.dtype)
        base = Uniform(
            torch.full_like(self.loc, finfo.tiny),
            torch.full_like(self.loc, 1.0 - finfo.eps),
            validate_args=False,
        )
        transforms = [SigmoidTransform().inv, AffineTransform(self.loc, self.scale)]
        super().__init__(base, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        return type(self)(
            self.loc.expand(batch_shape),
            self.scale.expand(batch_shape),
            validate_args=False,
        )

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return self.scale**2 * math.pi**2 / 3

    def cdf(self, value):
        return torch.sigmoid((value - self.loc) / self.scale)

    def icdf(self, p):
        return self.loc + self.scale * torch.logit(p)
