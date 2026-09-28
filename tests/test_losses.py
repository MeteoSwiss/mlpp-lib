import keras
import numpy as np
import pytest
import scoringrules as sr
import torch
from scipy import stats

from mlpp_lib import losses
from mlpp_lib.custom_distributions import (
    CensoredNormalDistribution,
    TruncatedNormalDistribution,
)
from mlpp_lib.exceptions import MissingReparameterizationError
from mlpp_lib.probabilistic_layers import DistributionLayer, WrappingTorchDist

BATCH, N_FEATURES = 64, 4

# closed-form losses and the distribution they apply to
CLOSED_FORM = {
    "CRPSNormal": ("Normal", {}),
    "CRPSLogistic": ("Logistic", {}),
    "CRPSLogNormal": ("LogNormal", {}),
    "CRPSTruncatedNormal": ("TruncatedNormal", {"low": 0.0}),
    "CRPSCensoredNormal": ("CensoredNormal", {"low": 0.0, "high": 1.0}),
    "CRPSExponential": ("Exponential", {}),
    "CRPSPoisson": ("Poisson", {}),
    "CRPSMixtureNormal": ("MixtureNormal", {"num_components": 2}),
}


def _predict(name, event_size=1, seed=0, **kwargs):
    keras.utils.set_random_seed(seed)
    layer = DistributionLayer(name, event_size=event_size, distribution_kwargs=kwargs)
    inputs = torch.randn(BATCH, N_FEATURES)
    return layer, layer(inputs)


def _targets(name, event_size=1):
    torch.manual_seed(1)
    if name == "Poisson":
        return torch.randint(0, 5, (BATCH, event_size)).float()
    if name == "Bernoulli":
        return torch.randint(0, 2, (BATCH, event_size)).float()
    return torch.rand(BATCH, event_size) * 0.8 + 0.1


def test_crps_normal_against_closed_form():
    mu, sigma = torch.randn(32, 1), torch.rand(32, 1) + 0.5
    y_true = torch.randn(32, 1)
    y_pred = WrappingTorchDist(torch.distributions.Normal(mu, sigma))
    loss = losses.CRPSNormal()(y_true, y_pred)

    z = (y_true - mu) / sigma
    expected = sigma * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    np.testing.assert_allclose(loss.item(), expected.mean().item(), rtol=1e-5)


@pytest.mark.parametrize("loss_name", CLOSED_FORM)
def test_closed_form_vs_monte_carlo(loss_name):
    """The closed-form CRPS must agree with its Monte Carlo estimate."""
    dist_name, kwargs = CLOSED_FORM[loss_name]
    _, y_pred = _predict(dist_name, event_size=2, **kwargs)
    y_true = _targets(dist_name, event_size=2)

    closed_form = getattr(losses, loss_name)()(y_true, y_pred)
    with torch.no_grad():
        monte_carlo = losses.CRPSEnsemble(num_samples=5000)(y_true, y_pred)
    np.testing.assert_allclose(closed_form.item(), monte_carlo.item(), rtol=2e-2)


@pytest.mark.parametrize("loss_name", CLOSED_FORM)
def test_closed_form_gradients(loss_name):
    dist_name, kwargs = CLOSED_FORM[loss_name]
    layer, y_pred = _predict(dist_name, event_size=2, **kwargs)
    y_true = _targets(dist_name, event_size=2)
    getattr(losses, loss_name)()(y_true, y_pred).backward()
    for weight in layer.trainable_weights:
        assert torch.isfinite(weight.value.grad).all()


@pytest.mark.parametrize(
    "loss",
    [
        losses.CRPSEnsemble(num_samples=10),
        losses.TWCRPSEnsemble(num_samples=10, a=0.5),
        losses.WeightedCRPSEnergy(threshold=0.5, num_samples=10),
        losses.NegativeLogLikelihood(),
    ],
    ids=lambda loss: type(loss).__name__,
)
@pytest.mark.parametrize("dist_name", ["Normal", "Gamma", "Beta", "Weibull"])
def test_sample_losses_gradients(loss, dist_name):
    layer, y_pred = _predict(dist_name, event_size=3)
    y_true = _targets(dist_name, event_size=3)
    value = loss(y_true, y_pred)
    assert value.ndim == 0
    value.backward()
    for weight in layer.trainable_weights:
        assert torch.isfinite(weight.value.grad).all()


def test_sample_loss_requires_rsample():
    _, y_pred = _predict("Poisson")
    y_true = _targets("Poisson")
    with pytest.raises(MissingReparameterizationError):
        losses.CRPSEnsemble()(y_true, y_pred)
    # fine without gradients, e.g. for evaluation
    with torch.no_grad():
        losses.CRPSEnsemble()(y_true, y_pred)


def test_energy_score():
    layer, y_pred = _predict("MultivariateNormalTriL", event_size=3)
    y_true = _targets("MultivariateNormalTriL", event_size=3)
    loss = losses.EnergyScore(num_samples=50)
    per_sample = losses.EnergyScore(num_samples=50, reduction="none")(y_true, y_pred)
    assert per_sample.shape == (BATCH,)
    loss(y_true, y_pred).backward()
    for weight in layer.trainable_weights:
        assert torch.isfinite(weight.value.grad).all()

    # in 1D, the energy score is the CRPS
    _, y_pred = _predict("Normal")
    y_true = _targets("Normal")
    with torch.no_grad():
        es = losses.EnergyScore(num_samples=3000)(y_true, y_pred)
    np.testing.assert_allclose(
        es.item(), losses.CRPSNormal()(y_true, y_pred).item(), rtol=3e-2
    )


def test_ensemble_tensor_predictions():
    y_true = torch.randn(BATCH, 2)
    ensemble = torch.randn(BATCH, 20, 2)
    loss = losses.CRPSEnsemble(estimator="nrg")(y_true, ensemble)
    expected = sr.crps_ensemble(
        y_true.numpy(), ensemble.numpy(), m_axis=1, estimator="nrg"
    ).mean()
    np.testing.assert_allclose(loss.item(), expected, rtol=1e-5)


def test_raw_torch_distribution():
    y_true = torch.randn(BATCH, 1)
    dist = torch.distributions.Normal(torch.zeros(BATCH, 1), torch.ones(BATCH, 1))
    np.testing.assert_allclose(
        losses.CRPSNormal()(y_true, dist).item(),
        losses.CRPSNormal()(y_true, WrappingTorchDist(dist)).item(),
    )


def test_reductions_and_sample_weights():
    _, y_pred = _predict("Normal", event_size=2)
    y_true = _targets("Normal", event_size=2)

    per_element = losses.CRPSNormal(reduction="none")(y_true, y_pred)
    assert per_element.shape == (BATCH, 2)
    np.testing.assert_allclose(
        losses.CRPSNormal()(y_true, y_pred).item(), per_element.mean().item(), rtol=1e-6
    )
    np.testing.assert_allclose(
        losses.CRPSNormal(reduction="sum")(y_true, y_pred).item(),
        per_element.sum().item(),
        rtol=1e-6,
    )

    sample_weight = torch.zeros(BATCH)
    sample_weight[: BATCH // 2] = 1.0
    weighted = losses.CRPSNormal(reduction="mean")(
        y_true, y_pred, sample_weight=sample_weight
    )
    np.testing.assert_allclose(
        weighted.item(), per_element[: BATCH // 2].mean().item(), rtol=1e-6
    )
    # [batch, 1] weights are supported too
    weighted_2d = losses.CRPSNormal(reduction="mean")(
        y_true, y_pred, sample_weight=sample_weight[:, None]
    )
    np.testing.assert_allclose(weighted.item(), weighted_2d.item())


@pytest.mark.parametrize(
    "loss",
    [
        losses.CRPSNormal(),
        losses.CRPSEnsemble(),
        losses.NegativeLogLikelihood(),
        losses.MultivariateLoss("mse", scaling="standard"),
    ],
    ids=lambda loss: type(loss).__name__,
)
def test_nan_targets_are_ignored(loss):
    layer, y_pred = _predict("Normal", event_size=2)
    y_true = _targets("Normal", event_size=2)
    y_nan = y_true.clone()
    y_nan[:5, 0] = torch.nan

    keras.utils.set_random_seed(2)
    value = loss(y_nan, y_pred)
    assert torch.isfinite(value)
    value.backward()
    for weight in layer.trainable_weights:
        assert torch.isfinite(weight.value.grad).all()

    if isinstance(loss, losses.CRPSNormal):
        per_element = losses.CRPSNormal(reduction="none")(y_true, y_pred)
        mask = torch.isfinite(y_nan)
        np.testing.assert_allclose(
            value.item(), per_element[mask].mean().item(), rtol=1e-6
        )


def test_negative_log_likelihood():
    _, y_pred = _predict("Gamma", event_size=2)
    y_true = _targets("Gamma", event_size=2)
    expected = -y_pred.log_prob(y_true).mean() / 2
    np.testing.assert_allclose(
        losses.NegativeLogLikelihood()(y_true, y_pred).item(),
        expected.item(),
        rtol=1e-6,
    )


def test_multivariate_loss():
    y_true = torch.randn(BATCH, 3) * torch.tensor([1.0, 10.0, 100.0])
    y_pred = y_true + torch.randn(BATCH, 3)

    mse = losses.MultivariateLoss("mse", reduction="none")(y_true, y_pred)
    np.testing.assert_allclose(mse, (y_true - y_pred) ** 2, rtol=1e-6)

    weights = [1.0, 0.0, 2.0]
    mae = losses.MultivariateLoss("mae", weights=weights, reduction="none")(
        y_true, y_pred
    )
    np.testing.assert_allclose(
        mae, torch.abs(y_true - y_pred) * torch.tensor(weights), rtol=1e-6
    )

    scaled = losses.MultivariateLoss("mse", scaling="standard", reduction="none")(
        y_true, y_pred
    )
    std = y_true.std(0, unbiased=False)
    np.testing.assert_allclose(
        scaled, ((y_true - y_pred) / std) ** 2, rtol=1e-4, atol=1e-6
    )

    losses.MultivariateLoss("mae", scaling="minmax")(y_true, y_pred)

    # with distributions
    _, dist = _predict("Normal", event_size=3)
    losses.MultivariateLoss("mse")(y_true, dist)
    losses.MultivariateLoss("crps_energy", scaling="minmax")(y_true, dist).backward()

    with pytest.raises(NotImplementedError):
        losses.MultivariateLoss("rmse")
    with pytest.raises(ValueError):
        losses.MultivariateLoss("mse", weights=[1.0])(y_true, y_pred)


def test_combined_loss():
    _, y_pred = _predict("Normal", event_size=2)
    y_true = _targets("Normal", event_size=2)
    combined = losses.CombinedLoss(
        [
            {"CRPSNormal": {}, "weight": 0.7},
            {"MultivariateLoss": {"metric": "mse"}, "weight": 0.3},
        ]
    )
    expected = 0.7 * losses.CRPSNormal()(
        y_true, y_pred
    ) + 0.3 * losses.MultivariateLoss("mse")(y_true, y_pred)
    np.testing.assert_allclose(combined(y_true, y_pred).item(), expected.item())


def test_weighted_crps_energy_legacy_arguments():
    with pytest.warns(DeprecationWarning):
        loss = losses.WeightedCRPSEnergy(
            threshold=0.1, n_samples=50, correct_crps=False
        )
    assert loss.num_samples == 50
    assert loss.estimator == "qd"
    assert loss.fn_kwargs == {"a": 0.1, "b": float("inf")}
    assert losses.WeightedCRPSEnergy(threshold=0.1).num_samples == 1000


def test_wrong_inputs():
    y_true = torch.randn(BATCH, 1)
    with pytest.raises(TypeError):
        losses.CRPSNormal()(y_true, torch.randn(BATCH, 1))
    _, y_pred = _predict("Weibull")
    with pytest.raises(ValueError, match="closed-form"):
        losses.CRPSNormal()(y_true, y_pred)
    with pytest.raises(ValueError):
        losses.CRPSEnsemble(num_samples=1)


def test_scoringrules_backend_is_restored():
    previous = sr.backends._active
    _, y_pred = _predict("CensoredNormal")
    losses.CRPSCensoredNormal()(_targets("CensoredNormal"), y_pred)
    assert sr.backends._active == previous


ALL_LOSSES = [
    losses.DistributionLossWrapper(fn=sr.crps_normal),
    losses.DistributionLossWrapper(fn="scoringrules.crps_normal", reduction="sum"),
    losses.SampleLossWrapper(fn=sr.twcrps_ensemble, num_samples=5, a=0.3),
    *[getattr(losses, name)() for name in CLOSED_FORM],
    losses.CRPSEnsemble(num_samples=7, estimator="fair"),
    losses.TWCRPSEnsemble(num_samples=7, a=0.1, b=0.9),
    losses.WeightedCRPSEnergy(threshold=0.3, num_samples=7),
    losses.EnergyScore(num_samples=7),
    losses.NegativeLogLikelihood(),
    losses.MultivariateLoss("mae", scaling="minmax", weights=[1.0]),
    losses.CombinedLoss([{"CRPSNormal": {}, "weight": 0.5}, "NegativeLogLikelihood"]),
]


@pytest.mark.parametrize("loss", ALL_LOSSES, ids=lambda loss: type(loss).__name__)
def test_serialization(loss):
    serialized = keras.saving.serialize_keras_object(loss)
    restored = keras.saving.deserialize_keras_object(serialized)
    assert type(restored) is type(loss)
    assert restored.get_config() == loss.get_config()


@pytest.mark.parametrize(
    "bounds",
    [(0.0, np.inf), (-np.inf, 1.0), (-1.0, 2.0), (0.0, 1.0), (-np.inf, np.inf)],
)
def test_crps_truncated_and_censored_normal(bounds):
    """Our stable implementations agree with scoringrules."""
    low, high = bounds
    rng = np.random.default_rng(0)
    obs, loc = rng.normal(size=(2, 100))
    scale = rng.uniform(0.3, 2.0, 100)
    args = [torch.tensor(a) for a in (obs, loc, scale)]
    np.testing.assert_allclose(
        losses.crps_truncated_normal(*args, low, high),
        sr.crps_tnormal(obs, loc, scale, low, high, backend="numpy"),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        losses.crps_censored_normal(*args, low, high),
        sr.crps_cnormal(obs, loc, scale, low, high, backend="numpy"),
        rtol=1e-6,
        atol=1e-8,
    )


def test_crps_truncated_and_censored_normal_tails():
    """Far from the bounds (where scoringrules gives NaN) in float32."""
    obs = torch.tensor([0.1, 0.05, 0.6])
    loc = torch.tensor([-5.0, -8.0, -1.26], requires_grad=True)
    scale = torch.tensor([0.5, 0.5, 0.0245], requires_grad=True)
    torch.manual_seed(0)
    for crps_fn, dist_cls, high in [
        (losses.crps_truncated_normal, TruncatedNormalDistribution, np.inf),
        (losses.crps_censored_normal, CensoredNormalDistribution, 1.0),
    ]:
        crps = crps_fn(obs, loc, scale, 0.0, high)
        crps.sum().backward()
        assert torch.isfinite(loc.grad).all() and torch.isfinite(scale.grad).all()
        dist = dist_cls(loc.detach().double(), scale.detach().double(), 0.0, high)
        mc = sr.crps_ensemble(
            obs.double().numpy(),
            dist.sample((200_000,)).numpy(),
            m_axis=0,
            estimator="pwm",
        )
        np.testing.assert_allclose(crps.detach(), mc, rtol=3e-2)
        loc.grad, scale.grad = None, None


# estimators comparing all pairs of samples need memory ~ batch * num_samples**2
PAIRWISE_ESTIMATORS = {"nrg", "fair"}


@pytest.mark.parametrize(
    "loss",
    [
        losses.CRPSEnsemble(),
        losses.TWCRPSEnsemble(a=0.0),
        losses.WeightedCRPSEnergy(threshold=0.0),
        losses.EnergyScore(),
        losses.SampleLossWrapper(fn=sr.crps_ensemble),
    ],
    ids=lambda loss: type(loss).__name__,
)
def test_default_estimators_are_not_pairwise(loss):
    assert loss.estimator not in PAIRWISE_ESTIMATORS
    with pytest.warns(DeprecationWarning):
        biased = losses.WeightedCRPSEnergy(threshold=0.0, correct_crps=False)
    assert biased.estimator not in PAIRWISE_ESTIMATORS


def test_energy_score_estimator_matches_fair():
    """The default (linear-memory) estimator agrees with the fair estimator."""
    _, y_pred = _predict("MultivariateNormalTriL", event_size=2)
    y_true = _targets("MultivariateNormalTriL", event_size=2)
    torch.manual_seed(0)
    with torch.no_grad():
        default = losses.EnergyScore(num_samples=2000)(y_true, y_pred)
        fair = losses.EnergyScore(num_samples=2000, estimator="fair")(y_true, y_pred)
    np.testing.assert_allclose(default.item(), fair.item(), rtol=2e-2)
