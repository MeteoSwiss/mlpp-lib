import math

import keras
import numpy as np
import pytest
import torch

from mlpp_lib.exceptions import MissingReparameterizationError
from mlpp_lib.probabilistic_layers import (
    DISTRIBUTIONS,
    LEGACY_DISTRIBUTIONS,
    BaseParametricDistribution,
    DistributionLayer,
    ParametricNormal,
    WrappingTorchDist,
    get_distribution,
)

BATCH, N_FEATURES = 16, 5
ALL_DISTRIBUTIONS = list(DISTRIBUTIONS)
RSAMPLE_DISTRIBUTIONS = [n for n, c in DISTRIBUTIONS.items() if c().has_rsample]
NO_RSAMPLE_DISTRIBUTIONS = [n for n in DISTRIBUTIONS if n not in RSAMPLE_DISTRIBUTIONS]


def _layer(name, event_size=1, **kwargs):
    layer = DistributionLayer(name, event_size=event_size, **kwargs)
    layer.build((None, N_FEATURES))
    return layer


def test_registry():
    assert len(DISTRIBUTIONS) >= 13
    for name, cls in DISTRIBUTIONS.items():
        assert issubclass(cls, BaseParametricDistribution)
        assert cls.name == name
    assert set(NO_RSAMPLE_DISTRIBUTIONS) == {"Poisson", "Bernoulli", "MixtureNormal"}


@pytest.mark.parametrize("event_size", [1, 3])
@pytest.mark.parametrize("name", ALL_DISTRIBUTIONS)
def test_output_types_and_shapes(name, event_size):
    layer = _layer(name, event_size)
    inputs = torch.randn(BATCH, N_FEATURES)

    assert layer.parameters_encoder.units == layer.distribution.num_parameters

    dist = layer(inputs)
    assert isinstance(dist, WrappingTorchDist)
    assert dist.batch_shape == (BATCH,)
    assert dist.event_shape == (event_size,)
    assert dist.mean.shape == (BATCH, event_size)

    expected = layer(inputs, output_type="expected")
    assert expected.shape == (BATCH, event_size)
    assert tuple(layer.compute_output_shape((BATCH, N_FEATURES))) == (
        BATCH,
        event_size,
    )

    samples = layer(inputs, output_type="samples", num_samples=7)
    assert samples.shape == (7, BATCH, event_size)
    samples = layer(inputs, output_type="samples", num_samples=7, pattern="bsd")
    assert samples.shape == (BATCH, 7, event_size)
    # default number of samples
    samples = layer(inputs, output_type="samples")
    assert samples.shape == (layer.num_samples, BATCH, event_size)

    # log_prob is forwarded to the distribution and is finite on its samples
    log_prob = dist.log_prob(dist.sample(3))
    assert log_prob.shape == (3, BATCH)
    assert torch.isfinite(log_prob).all()


@pytest.mark.parametrize("name", RSAMPLE_DISTRIBUTIONS)
def test_gradient_flow_through_samples(name):
    layer = _layer(name, event_size=2)
    inputs = torch.randn(BATCH, N_FEATURES)
    samples = layer(inputs, output_type="samples", num_samples=5, training=True)
    samples.mean().backward()
    for weight in layer.trainable_weights:
        assert weight.value.grad is not None
        assert torch.isfinite(weight.value.grad).all()


@pytest.mark.parametrize("name", NO_RSAMPLE_DISTRIBUTIONS)
def test_defense_missing_rsample(name):
    layer = _layer(name)
    inputs = torch.randn(BATCH, N_FEATURES)
    # sampling in training mode must fail, instead of silently having no gradients
    with pytest.raises(MissingReparameterizationError):
        layer(inputs, output_type="samples", training=True)
    with pytest.raises(MissingReparameterizationError):
        layer(inputs).rsample(3)
    # it is fine at inference time
    assert layer(inputs, output_type="samples", num_samples=3).shape == (3, BATCH, 1)


def test_positive_parameters():
    layer = _layer("Normal", event_size=2)
    # large negative inputs would give a zero scale without the floor
    dist = layer(torch.full((BATCH, N_FEATURES), -1e4))
    assert (dist.base_distribution.scale > 0).all()


def test_multivariate_normal_tril():
    layer = _layer("MultivariateNormalTriL", event_size=4)
    assert layer.distribution.num_parameters == 4 + 10
    dist = layer(torch.randn(BATCH, N_FEATURES))
    scale_tril = dist.scale_tril
    assert scale_tril.shape == (BATCH, 4, 4)
    assert torch.allclose(scale_tril, torch.tril(scale_tril))
    assert (torch.diagonal(scale_tril, dim1=-2, dim2=-1) > 0).all()
    # valid covariance matrices
    torch.linalg.cholesky(dist.covariance_matrix)


def test_truncated_and_censored_bounds():
    inputs = torch.randn(BATCH, N_FEATURES)
    for name in ["TruncatedNormal", "CensoredNormal"]:
        layer = _layer(name, distribution_kwargs={"low": -1.0, "high": 2.0})
        samples = layer(inputs, output_type="samples", num_samples=100)
        assert samples.min() >= -1.0 and samples.max() <= 2.0
        # default is [0, inf]
        layer = _layer(name)
        samples = layer(inputs, output_type="samples", num_samples=100)
        assert samples.min() >= 0.0
    with pytest.raises(ValueError):
        get_distribution("TruncatedNormal", low=1.0, high=0.0)


def test_bias_init_array():
    bias = np.array([1.0, 2.0, 3.0])
    layer = _layer("Normal", event_size=3, bias_init=bias)
    encoder_bias = keras.ops.convert_to_numpy(layer.parameters_encoder.bias)
    np.testing.assert_allclose(encoder_bias, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

    layer = _layer("MultivariateNormalTriL", event_size=3, bias_init=bias)
    encoder_bias = keras.ops.convert_to_numpy(layer.parameters_encoder.bias)
    np.testing.assert_allclose(encoder_bias[:3], bias)
    assert encoder_bias.shape == (3 + 6,)

    with pytest.raises(ValueError):
        _layer("Normal", event_size=1, bias_init=np.zeros(3))


@pytest.mark.parametrize(
    "bias_init", ["zeros", np.array([1.0, 2.0]), keras.initializers.Constant(0.5)]
)
def test_layer_config_roundtrip(bias_init):
    layer = _layer(
        "TruncatedNormal",
        event_size=2,
        distribution_kwargs={"low": -1.0, "high": math.inf},
        num_samples=11,
        bias_init=bias_init,
    )
    config = layer.get_config()
    # the config must be json-serializable (no pickled torch modules)
    keras.saving.serialize_keras_object(layer)
    restored = DistributionLayer.from_config(config)
    assert restored.distribution.get_config() == {
        "event_size": 2,
        "low": -1.0,
        "high": math.inf,
    }
    assert restored.num_samples == 11
    restored.build((None, N_FEATURES))
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(restored.parameters_encoder.bias),
        keras.ops.convert_to_numpy(layer.parameters_encoder.bias),
    )


def test_mixture_normal_config():
    distribution = get_distribution("MixtureNormal", event_size=2, num_components=3)
    assert distribution.num_parameters == 3 * 3 * 2
    layer = DistributionLayer(distribution)
    restored = DistributionLayer.from_config(layer.get_config())
    assert restored.distribution.num_components == 3


@pytest.mark.parametrize("legacy_name", list(LEGACY_DISTRIBUTIONS))
def test_legacy_names(legacy_name):
    new_name, options = LEGACY_DISTRIBUTIONS[legacy_name]
    with pytest.warns(DeprecationWarning, match=legacy_name):
        distribution = get_distribution(legacy_name, event_size=2)
    assert distribution.name == new_name
    assert distribution.event_size == 2
    for key, value in options.items():
        assert getattr(distribution, key) == value


def test_legacy_options_are_ignored():
    with pytest.warns(DeprecationWarning, match="convert_to_tensor_fn"):
        distribution = get_distribution(
            "IndependentNormal", event_shape=(1,), convert_to_tensor_fn="mean"
        )
    assert isinstance(distribution, ParametricNormal)


def test_errors():
    with pytest.raises(KeyError, match="not available"):
        get_distribution("NotADistribution")
    with pytest.raises(TypeError):
        DistributionLayer(torch.distributions.Normal)
    layer = _layer("Normal")
    with pytest.raises(ValueError, match="output_type"):
        layer(torch.randn(BATCH, N_FEATURES), output_type="quantiles")
    with pytest.raises(ValueError, match="pattern"):
        layer(torch.randn(BATCH, N_FEATURES)).sample(2, pattern="dsb")


def test_wrapping_torch_dist():
    dist = WrappingTorchDist(
        torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(4, 2), torch.ones(4, 2)), 1
        )
    )
    assert dist.name == "Normal"
    assert isinstance(dist.base_distribution, torch.distributions.Normal)
    assert dist.variance.shape == (4, 2)
    assert dist.sample((3, 5)).shape == (3, 5, 4, 2)
    assert dist.sample((3, 5), pattern="bsd").shape == (4, 3, 5, 2)
    assert "Normal" in repr(dist)
    with pytest.raises(AttributeError):
        dist.not_an_attribute
