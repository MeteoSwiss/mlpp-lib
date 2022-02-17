from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def crps_nrg_ensemble(
    fct_ensemble: Union[tf.Tensor, np.ndarray], obs: Union[tf.Tensor, np.ndarray]
) -> tf.Tensor:
    """
    Energy form of the Continuous Ranked Probability Score from Gneiting and Raftery (2007),
    where the expectations are calculated from an ensemble.

    .. math::
        CRPS(F, y) = E_F|X - y| - 1/2 * E_F|X - X'|

    """

    # first term
    diff = tf.abs(fct_ensemble - obs[None, :])
    diff = tf.reduce_mean(diff, axis=0)

    # second term
    forecasts_diff = tf.abs(fct_ensemble[None, :] - fct_ensemble[:, None])
    forecasts_diff = 0.5 * tf.reduce_mean(forecasts_diff, axis=0)

    return diff - forecasts_diff


def crps_nrg(
    fct_dist: tfp.distributions.Distribution, obs: Union[tf.Tensor, np.ndarray]
) -> tf.Tensor:
    """
    Energy form of the Continuous Ranked Probability Score from Gneiting and Raftery (2007),
    where the expectations are calculated from the distribution itself.

    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    """

    n_samples = 10000

    obs = tf.debugging.check_numerics(obs, "Target values")

    # first term
    E_1 = tfp.monte_carlo.expectation(
        f=lambda x: tf.abs(x - obs[None, :]), samples=fct_dist.sample(n_samples)
    )

    # second term
    E_2 = tfp.monte_carlo.expectation(
        f=lambda x: tf.abs(x[0] - x[1]),
        samples=[fct_dist.sample(n_samples), fct_dist.sample(n_samples)],
    )

    crps = E_1 - E_2 / 2

    return crps
