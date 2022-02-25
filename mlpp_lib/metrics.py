from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def crps_energy_ensemble(
    obs: Union[tf.Tensor, np.ndarray],
    fct_ens: Union[tf.Tensor, np.ndarray],
) -> tf.Tensor:
    """
    Energy form of the Continuous Ranked Probability Score from Gneiting and Raftery (2007),
    where the expectations terms are approximated from an ensemble.

    .. math::
        CRPS(F, y) = E_F|X - y| - 1/2 * E_F|X - X'|

    Parameters
    ----------
    fct_ens: array-like
        Ensemble forecasts, with ensemble members along the first dimension.
    obs: array-like
        Observations.

    Return
    ------
    crps: tf.Tensor
        The CRPS for each sample.
    """

    assert fct_ens.shape[1:] == obs.shape

    # first term
    E_1 = tf.abs(fct_ens - obs[None, :])
    E_1 = tf.reduce_mean(E_1, axis=0)

    # second term
    E_2 = tf.abs(fct_ens[None, :] - fct_ens[:, None])
    E_2 = tf.reduce_mean(E_2, axis=(0, 1))
    crps = E_1 - E_2 / 2

    return crps


def crps_energy(
    obs: Union[tf.Tensor, np.ndarray],
    fct_dist: tfp.distributions.Distribution,
) -> tf.Tensor:
    """
    Energy form of the Continuous Ranked Probability Score from Gneiting and Raftery (2007),
    where the expectation terms are approximated from the distribution using monte-carlo methods.

    .. math::
        CRPS(F, y) = E_F|X - y| - 1/2 * E_F|X - X'|

    Parameters
    ----------
    obs: array-like
        Array of observations.
    fct_dist: tensorflow-probability Distribution
        The predicted distribution.

    Return
    ------
    crps: tf.Tensor
        The CRPS for each sample.
    """

    assert fct_dist.batch_shape == obs.shape

    n_samples = 1000

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
