import numpy as np
import pytest
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from mlpp_lib import losses


def test_crps_energy():

    tf.random.set_seed(1234)
    fct_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    fct_dist = tfd.Independent(fct_dist, reinterpreted_batch_ndims=1)
    fct_dist.shape = (*fct_dist.batch_shape, *fct_dist.event_shape)
    obs = tf.zeros((3, 1))

    result = losses.crps_energy(obs, fct_dist)
    good_result = tf.constant([[0.284208], [0.280653], [0.206382]])

    np.testing.assert_allclose(result, good_result, atol=1e-5)


def test_crps_energy_ensemble():

    tf.random.set_seed(1234)
    fct_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    fct_ensemble = fct_dist.sample(100)
    obs = tf.zeros((3, 1))

    result = losses.crps_energy_ensemble(obs, fct_ensemble)
    good_result = tf.constant([[0.22706872], [0.22341758], [0.21489006]])

    np.testing.assert_allclose(result, good_result, atol=1e-5)


def test_weighted_crps():

    tf.random.set_seed(1234)

    threshold = 1
    loss = losses.WeightedCRPSEnergy(threshold)

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


def test_weighted_crps_zero_sample_weights():
    """Passing an array of all zeros as sample weights set the total loss to zero"""
    loss = losses.WeightedCRPSEnergy(threshold=0)
    tf.random.set_seed(1234)
    fct_dist = tfd.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    obs = tf.zeros((3, 1))
    sample_weights = tf.zeros((3, 1))
    result = loss(obs, fct_dist, sample_weight=sample_weights)
    assert result == 0


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
