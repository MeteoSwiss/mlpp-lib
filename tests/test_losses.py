from inspect import getmembers, isclass

import numpy as np
import pytest
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from mlpp_lib import losses
from mlpp_lib import probabilistic_layers


LAYERS = [obj[0] for obj in getmembers(probabilistic_layers, isclass)]


def crps_closed_form_gaussian(obs, mu, sigma):
    loc = (obs - mu) / sigma
    phi = 1.0 / np.sqrt(2.0 * np.pi) * tf.math.exp(-tf.math.square(loc) / 2.0)
    Phi = 0.5 * (1.0 + tf.math.erf(loc / np.sqrt(2.0)))
    crps_closed_form = tf.math.sqrt(tf.math.square(sigma)) * (
        loc * (2.0 * Phi - 1.0) + 2 * phi - 1.0 / tf.math.sqrt(np.pi)
    )
    return crps_closed_form


def test_crps_energy():
    batch_size = 100
    tf.random.set_seed(1234)
    mu = tf.zeros((batch_size, 1))
    sigma = tf.ones((batch_size, 1))
    fct_dist = tfd.Normal(loc=mu, scale=sigma)
    fct_dist = tfd.Independent(fct_dist, reinterpreted_batch_ndims=1)
    fct_dist.shape = fct_dist.batch_shape + fct_dist.event_shape
    obs = tf.zeros((batch_size, 1))

    result = losses.crps_energy(obs, fct_dist)
    good_result = crps_closed_form_gaussian(obs, mu, sigma)

    np.testing.assert_allclose(
        tf.reduce_mean(result), tf.reduce_mean(good_result), atol=1e-2
    )


def test_crps_energy_ensemble():
    batch_size = 100
    tf.random.set_seed(1234)
    mu = tf.zeros((batch_size, 1))
    sigma = tf.ones((batch_size, 1))
    fct_dist = tfd.Normal(loc=mu, scale=sigma)
    fct_ensemble = fct_dist.sample(1000)
    obs = tf.zeros((batch_size, 1))

    result = losses.crps_energy_ensemble(obs, fct_ensemble)
    good_result = crps_closed_form_gaussian(obs, mu, sigma)

    np.testing.assert_allclose(
        tf.reduce_mean(result), tf.reduce_mean(good_result), atol=1e-2
    )


@pytest.mark.parametrize("layer", LAYERS)
def test_weighted_crps_layers(layer):
    event_shape = (1,)
    batch_shape = (10,)
    event_size = event_shape[0]
    layer_class = getattr(probabilistic_layers, layer)
    prob_layer = layer_class(event_size)
    y_pred_dist = prob_layer(
        np.random.random(batch_shape + (layer_class.params_size(event_size),))
    )
    loss = losses.WeightedCRPSEnergy(threshold=0, reduction="none")
    result = loss(tf.zeros(batch_shape + event_shape), y_pred_dist)
    assert result.shape == batch_shape + event_shape


def test_weighted_crps_dtypes():
    """Test various input data types"""

    tf.random.set_seed(1234)
    loss = losses.WeightedCRPSEnergy(threshold=1)
    y_pred_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    y_pred_ens = y_pred_dist.sample(100)
    y_true = tf.zeros((3, 1))

    # prediction is TFP distribution
    tf.random.set_seed(42)
    result = loss(y_true, y_pred_dist)
    assert tf.is_tensor(result)
    assert result.dtype == "float32"
    tf.random.set_seed(42)
    np.testing.assert_allclose(result, loss(y_true.numpy(), y_pred_dist))

    # prediction is TF tensor
    result = loss(y_true, y_pred_ens)
    assert tf.is_tensor(result)
    assert result.dtype == "float32"
    np.testing.assert_allclose(result, loss(y_true.numpy(), y_pred_ens))

    # prediction is numpy array
    result = loss(y_true, y_pred_ens.numpy())
    assert tf.is_tensor(result)
    assert result.dtype == "float32"
    np.testing.assert_allclose(result, loss(y_true.numpy(), y_pred_ens.numpy()))


def test_weighted_crps_high_threshold():
    """Using a very large threshold should set the loss to zero"""
    tf.random.set_seed(1234)
    loss = losses.WeightedCRPSEnergy(threshold=1e6)
    fct_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    obs = tf.zeros((3, 1))
    result = loss(obs, fct_dist)
    assert result == 1e-7


def test_weighted_crps_no_reduction():
    """Passing reduction='none' should return a loss value per sample"""
    tf.random.set_seed(1234)
    loss = losses.WeightedCRPSEnergy(threshold=0, reduction="none")
    fct_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    obs = tf.zeros((3, 1))
    result = loss(obs, fct_dist)
    assert result.shape == obs.shape


def test_weighted_crps_zero_sample_weights():
    """Passing an array of all zeros as sample weights set the total loss to zero"""
    tf.random.set_seed(1234)
    loss = losses.WeightedCRPSEnergy(threshold=0)
    fct_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    obs = tf.zeros((3, 1))
    sample_weights = tf.zeros((3, 1))
    result = loss(obs, fct_dist, sample_weight=sample_weights)
    assert result == 0


def test_multiscale_crps_layer():
    event_shape = (1,)
    batch_shape = (10,)
    event_size = event_shape[0]
    layer_class = getattr(probabilistic_layers, "IndependentGamma")
    prob_layer = layer_class(event_size)
    y_pred_dist = prob_layer(
        np.random.random(batch_shape + (layer_class.params_size(event_size),))
    )
    loss = losses.MultiScaleCRPSEnergy(threshold=0, scales=[1, 2], reduction="none")
    result = loss(tf.zeros(batch_shape + event_shape), y_pred_dist)
    assert result.shape == batch_shape + event_shape


def test_multiscale_crps_array():
    event_shape = (1,)
    batch_shape = (10,)
    event_size = event_shape[0]
    layer_class = getattr(probabilistic_layers, "IndependentGamma")
    prob_layer = layer_class(event_size)
    y_pred_dist = prob_layer(
        np.random.random(batch_shape + (layer_class.params_size(event_size),))
    )
    y_pred = y_pred_dist.sample(3)
    loss = losses.MultiScaleCRPSEnergy(threshold=0, scales=[1, 2], reduction="none")
    result = loss(tf.zeros(batch_shape + event_shape), y_pred)
    assert result.shape == batch_shape + event_shape


def test_energy_score():
    n_events, n_dims = 10, 3
    loss = losses.EnergyScore(reduction=tf.keras.losses.Reduction.NONE)
    fct_dist = tfd.MultivariateNormalDiag(
        loc=tf.zeros((n_events, n_dims)),
        scale_diag=tf.ones((n_events, n_dims)),
    )
    obs = tf.zeros((n_events, n_dims))
    result = loss(obs, fct_dist)
    assert tf.is_tensor(result)
    assert result.dtype == "float32"
    assert result.shape == obs.shape[0]


@pytest.mark.parametrize(
    "metric, scaling, weights",
    (
        ["mae", "standard", None],
        ["crps_energy", "minmax", None],
        ["mse", None, [1.0, 1.5]],
    ),
)
def test_multivariate_loss(metric, scaling, weights):

    tf.random.set_seed(1234)
    loss = losses.MultivariateLoss(metric, scaling, weights)

    if getattr(loss.metric, "loss_type", None) == "probabilistic":
        dist = tfd.Normal(loc=tf.zeros((3, 2)), scale=tf.ones((3, 2)))
        fct = tfd.Independent(dist, reinterpreted_batch_ndims=1)
        fct.shape = (*fct.batch_shape, *fct.event_shape)
    else:
        fct = tf.random.normal((3, 2))

    obs = tf.random.normal((3, 2))

    result = loss(obs, fct).numpy()

    assert isinstance(result, np.float32)


def test_binary_loss_dtypes():
    """Test various input data types"""
    tf.random.set_seed(1234)
    loss = losses.BinaryClassifierLoss(threshold=1)
    y_pred_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    y_pred_ens = y_pred_dist.sample(100)
    y_true = tf.zeros((3, 1))

    # prediction is TFP distribution
    tf.random.set_seed(42)
    result = loss(y_true, y_pred_dist)
    assert tf.is_tensor(result)
    assert result.dtype == "float32"
    tf.random.set_seed(42)
    np.testing.assert_allclose(result, loss(y_true.numpy(), y_pred_dist))

    # prediction is TF tensor
    result = loss(y_true, y_pred_ens)
    assert tf.is_tensor(result)
    assert result.dtype == "float32"
    np.testing.assert_allclose(result, loss(y_true.numpy(), y_pred_ens))

    # prediction is numpy array
    result = loss(y_true, y_pred_ens.numpy())
    assert tf.is_tensor(result)
    assert result.dtype == "float32"
    np.testing.assert_allclose(result, loss(y_true.numpy(), y_pred_ens.numpy()))


def test_combined_loss():
    """"""
    loss_specs = [
        {"BinaryClassifierLoss": {"threshold": 1}, "weight": 0.7},
        {"WeightedCRPSEnergy": {"threshold": 0.1}, "weight": 0.1},
    ]

    combined_loss = losses.CombinedLoss(loss_specs)
    y_pred_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    y_true = tf.zeros((3, 1))

    # prediction is TFP distribution
    tf.random.set_seed(42)
    result = combined_loss(y_true, y_pred_dist)
    assert tf.is_tensor(result)
    assert result.dtype == "float32"
