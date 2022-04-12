import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from mlpp_lib import losses


def test_crps_energy():

    tf.random.set_seed(1234)
    fct_dist = tfp.distributions.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    fct_dist = tfp.distributions.Independent(fct_dist, reinterpreted_batch_ndims=1)
    fct_dist.shape = (*fct_dist.batch_shape, *fct_dist.event_shape)
    obs = tf.zeros((3, 1))

    result = losses.crps_energy(obs, fct_dist)
    good_result = tf.constant([[0.284208], [0.280653], [0.206382]])

    np.testing.assert_allclose(result, good_result, atol=1e-5)


def test_crps_energy_ensemble():

    tf.random.set_seed(1234)
    fct_dist = tfp.distributions.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    fct_ensemble = fct_dist.sample(100)
    obs = tf.zeros((3, 1))

    result = losses.crps_energy_ensemble(obs, fct_ensemble)
    good_result = tf.constant([[0.22706872], [0.22341758], [0.21489006]])

    np.testing.assert_allclose(result, good_result, atol=1e-5)


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
        dist = tfp.distributions.Normal(loc=tf.zeros((3, 2)), scale=tf.ones((3, 2)))
        fct = tfp.distributions.Independent(dist, reinterpreted_batch_ndims=1)
        fct.shape = (*fct.batch_shape, *fct.event_shape)
    else:
        fct = tf.random.normal((3, 2))

    obs = tf.random.normal((3, 2))

    result = loss(obs, fct).numpy()

    assert isinstance(result, np.float32)
