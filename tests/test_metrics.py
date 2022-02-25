import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from mlpp_lib import metrics


def test_crps_energy():

    tf.random.set_seed(1234)
    fct_dist = tfp.distributions.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    obs = tf.zeros((3, 1))

    result = metrics.crps_energy(obs, fct_dist)
    good_result = tf.constant([[0.284208], [0.280653], [0.206382]])

    np.testing.assert_allclose(result, good_result, atol=1e-5)


def test_crps_energy_ensemble():

    tf.random.set_seed(1234)
    fct_dist = tfp.distributions.Normal(loc=tf.zeros((3, 1)), scale=tf.ones((3, 1)))
    fct_ensemble = fct_dist.sample(100)
    obs = tf.zeros((3, 1))

    result = metrics.crps_energy_ensemble(obs, fct_ensemble)
    good_result = tf.constant([[0.22706872], [0.22341758], [0.21489006]])

    np.testing.assert_allclose(result, good_result, atol=1e-5)
