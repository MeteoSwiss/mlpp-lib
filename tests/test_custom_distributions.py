import math

import numpy as np
import pytest
import torch
from scipy import stats

from mlpp_lib.custom_distributions import (
    CensoredNormalDistribution,
    LogisticDistribution,
    TruncatedNormalDistribution,
)

INF = math.inf

BOUNDS = [
    (0.0, INF),
    (-INF, 1.0),
    (-1.0, 2.0),
    (4.0, 6.0),
    (-INF, INF),
    (8.0, 9.0),  # far in the upper tail
    (-9.0, -8.5),  # far in the lower tail
]
BOUNDS_IDS = [f"[{a}, {b}]" for a, b in BOUNDS]


def _params(dtype=torch.float64):
    loc = torch.tensor([0.5, 1.0], dtype=dtype, requires_grad=True)
    scale = torch.tensor([1.0, 0.7], dtype=dtype, requires_grad=True)
    return loc, scale


def _scipy_truncnorm(loc, scale, low, high):
    loc, scale = loc.detach().numpy(), scale.detach().numpy()
    return stats.truncnorm(
        (low - loc) / scale, (high - loc) / scale, loc=loc, scale=scale
    )


def _all_finite(*tensors):
    return all(torch.isfinite(t).all() for t in tensors)


@pytest.mark.parametrize("bounds", BOUNDS, ids=BOUNDS_IDS)
def test_truncated_normal_moments(bounds):
    low, high = bounds
    loc, scale = _params()
    tn = TruncatedNormalDistribution(loc, scale, low, high)
    ref = _scipy_truncnorm(loc, scale, low, high)

    np.testing.assert_allclose(tn.mean.detach(), ref.mean(), rtol=1e-6)
    np.testing.assert_allclose(tn.variance.detach(), ref.var(), rtol=1e-5)
    np.testing.assert_allclose(tn.moment(1).detach(), ref.mean(), rtol=1e-6)
    np.testing.assert_allclose(tn.moment(2).detach(), ref.moment(2), rtol=1e-6)
    np.testing.assert_allclose(tn.moment(3).detach(), ref.moment(3), rtol=1e-5)

    # gradients must be finite, also for infinite bounds
    (tn.mean.sum() + tn.variance.sum() + tn.moment(3).sum()).backward()
    assert _all_finite(loc.grad, scale.grad)


@pytest.mark.parametrize("bounds", BOUNDS, ids=BOUNDS_IDS)
def test_truncated_normal_cdf_icdf_log_prob(bounds):
    low, high = bounds
    loc, scale = _params()
    tn = TruncatedNormalDistribution(loc, scale, low, high)
    ref = _scipy_truncnorm(loc, scale, low, high)

    p = np.array([0.1, 0.8])
    x = ref.ppf(p)
    np.testing.assert_allclose(
        tn.cdf(torch.tensor(x)).detach(), ref.cdf(x), rtol=1e-6, atol=1e-10
    )
    np.testing.assert_allclose(tn.icdf(torch.tensor(p)).detach(), x, rtol=1e-6)
    np.testing.assert_allclose(
        tn.log_prob(torch.tensor(x)).detach(), ref.logpdf(x), rtol=1e-6
    )
    # outside the support
    if math.isfinite(low):
        tn = TruncatedNormalDistribution(loc, scale, low, high, validate_args=False)
        assert tn.log_prob(torch.tensor([low - 1.0, low - 1.0]))[0] == -INF


@pytest.mark.parametrize("bounds", BOUNDS, ids=BOUNDS_IDS)
def test_truncated_normal_rsample(bounds):
    torch.manual_seed(0)
    low, high = bounds
    loc, scale = _params()
    tn = TruncatedNormalDistribution(loc, scale, low, high)
    ref = _scipy_truncnorm(loc, scale, low, high)

    samples = tn.rsample((100_000,))
    assert samples.shape == (100_000, 2)
    assert ((samples >= low) & (samples <= high)).all()
    np.testing.assert_allclose(samples.mean(0).detach(), ref.mean(), atol=2e-2)
    samples.mean().backward()
    assert _all_finite(loc.grad, scale.grad)

    assert not tn.sample((10,)).requires_grad


@pytest.mark.parametrize("bounds", BOUNDS[:5], ids=BOUNDS_IDS[:5])
def test_truncated_normal_float32(bounds):
    low, high = bounds
    loc, scale = _params(torch.float32)
    tn = TruncatedNormalDistribution(loc, scale, low, high)
    ref = _scipy_truncnorm(loc, scale, low, high)
    np.testing.assert_allclose(tn.mean.detach(), ref.mean(), rtol=1e-4)
    np.testing.assert_allclose(tn.variance.detach(), ref.var(), rtol=1e-3)
    samples = tn.rsample((1000,))
    assert ((samples >= low) & (samples <= high)).all()


@pytest.mark.parametrize("bounds", BOUNDS[:6], ids=BOUNDS_IDS[:6])
def test_censored_normal(bounds):
    torch.manual_seed(0)
    low, high = bounds
    loc, scale = _params()
    cn = CensoredNormalDistribution(loc, scale, low, high)

    samples = np.clip(
        stats.norm(loc.detach().numpy(), scale.detach().numpy()).rvs(
            (400_000, 2), random_state=0
        ),
        low,
        high,
    )
    np.testing.assert_allclose(cn.mean.detach(), samples.mean(0), atol=1e-2)
    np.testing.assert_allclose(cn.variance.detach(), samples.var(0), atol=2e-2)

    rsamples = cn.rsample((100_000,))
    assert ((rsamples >= low) & (rsamples <= high)).all()
    (cn.mean.sum() + cn.variance.sum() + rsamples.mean()).backward()
    assert _all_finite(loc.grad, scale.grad)


def test_censored_normal_log_prob_and_cdf():
    loc, scale = torch.tensor(0.0), torch.tensor(1.0)
    cn = CensoredNormalDistribution(loc, scale, low=-1.0, high=2.0)
    ref = stats.norm(0.0, 1.0)

    # point masses at the bounds
    np.testing.assert_allclose(
        cn.log_prob(torch.tensor(-1.0)), ref.logcdf(-1.0), rtol=1e-6
    )
    np.testing.assert_allclose(
        cn.log_prob(torch.tensor(2.0)), ref.logsf(2.0), rtol=1e-6
    )
    # density inside
    np.testing.assert_allclose(
        cn.log_prob(torch.tensor(0.5)), ref.logpdf(0.5), rtol=1e-6
    )

    np.testing.assert_allclose(cn.cdf(torch.tensor(-1.5)), 0.0)
    np.testing.assert_allclose(cn.cdf(torch.tensor(0.5)), ref.cdf(0.5), rtol=1e-6)
    np.testing.assert_allclose(cn.cdf(torch.tensor(2.0)), 1.0)


def test_logistic():
    loc = torch.tensor([0.0, 1.0])
    scale = torch.tensor([1.0, 2.0])
    dist = LogisticDistribution(loc, scale)
    ref = stats.logistic(loc.numpy(), scale.numpy())

    x = torch.tensor([0.3, -1.0])
    np.testing.assert_allclose(dist.log_prob(x), ref.logpdf(x.numpy()), rtol=1e-5)
    np.testing.assert_allclose(dist.cdf(x), ref.cdf(x.numpy()), rtol=1e-5)
    np.testing.assert_allclose(dist.mean, ref.mean(), rtol=1e-6)
    np.testing.assert_allclose(dist.variance, ref.var(), rtol=1e-6)
    np.testing.assert_allclose(
        dist.icdf(torch.tensor([0.2, 0.9])), ref.ppf([0.2, 0.9]), rtol=1e-5
    )

    torch.manual_seed(0)
    samples = dist.rsample((200_000,))
    assert torch.isfinite(samples).all()
    np.testing.assert_allclose(samples.mean(0), ref.mean(), atol=3e-2)


@pytest.mark.parametrize(
    "dist_cls", [TruncatedNormalDistribution, CensoredNormalDistribution]
)
def test_batch_shape_and_expand(dist_cls):
    dist = dist_cls(torch.zeros(4, 3), torch.ones(4, 3), 0.0, 1.0)
    assert dist.batch_shape == (4, 3)
    assert dist.rsample((5,)).shape == (5, 4, 3)
    expanded = dist.expand((2, 4, 3))
    assert expanded.batch_shape == (2, 4, 3)
    independent = torch.distributions.Independent(dist, 1)
    assert independent.event_shape == (3,)
    assert independent.log_prob(torch.full((4, 3), 0.5)).shape == (4,)
