from typing import Literal, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from mlpp_lib.decorators import with_attrs


@with_attrs(loss_type="probabilistic")
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

    # first term
    E_1 = tf.abs(fct_ens - obs[None, :])
    E_1 = tf.reduce_mean(E_1, axis=0)

    # second term
    E_2 = tf.abs(fct_ens[None, :] - fct_ens[:, None])
    E_2 = tf.reduce_mean(E_2, axis=(0, 1))
    crps = E_1 - E_2 / 2

    return crps


@with_attrs(loss_type="probabilistic")
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


class WeightedCRPSEnergy(tf.keras.losses.Loss):
    """
    Compute threshold-weighted CRPS using its kernel score representation.

    Parameters
    ----------
    threshold: float
        The threshold to be used within the weight function of the threshold-weighted CRPS.
    n_samples: int
        Number of samples used to compute the Monte Carlo expectations.
    correct_crps: bool
        Whether to bias correct the CRPS following Eq. 4 in Fricker et al. (2013).
    **kwargs:
        (Optional) Additional keyword arguments to be passed to the parent `Loss` class.

    Notes
    -----
    Currently only weight function w(x) = 1{x > t} is permitted, where t is a threshold of interest.
    """

    def __init__(
        self,
        threshold: float,
        n_samples: int = 1000,
        correct_crps: bool = True,
        **kwargs,
    ) -> None:
        super(WeightedCRPSEnergy, self).__init__(**kwargs)

        self.threshold = float(threshold)
        self.n_samples = int(n_samples)
        self.bias_correction = n_samples / (n_samples - 1) if correct_crps else 1.0

    def get_config(self) -> None:
        custom_config = {
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "bias_correction": self.bias_correction,
        }
        config = super().get_config()
        config.update(custom_config)
        return config

    def call(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: Union[tf.Tensor, np.ndarray, tfp.distributions.Distribution],
    ) -> tf.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        y_true: array-like
            Values representing the ground truth.
        y_pred: array_like or tfp.Distribution
            Predicted values or distributions.
        """

        threshold = tf.constant(self.threshold, dtype=y_true.dtype)
        y_true = tf.debugging.check_numerics(y_true, "Target values")
        v_obs = tf.math.maximum(y_true, threshold)

        if tf.is_tensor(y_pred) or isinstance(y_pred, np.ndarray):

            v_ens = tf.math.maximum(y_pred, threshold)

            # first term
            E_1 = tf.abs(v_ens - v_obs[None, :])

            # second term
            E_2 = tf.abs(v_ens[None, :] - v_ens[:, None])
            E_2 = tf.reduce_mean(E_2, axis=(0, 1))

        else:

            # first term
            E_1 = tfp.monte_carlo.expectation(
                f=lambda x: tf.abs(x - v_obs[None, :]),
                samples=tf.math.maximum(y_pred.sample(self.n_samples), threshold),
            )

            # second term
            E_2 = tfp.monte_carlo.expectation(
                f=lambda x: tf.abs(x[0] - x[1]),
                samples=[
                    tf.math.maximum(y_pred.sample(self.n_samples), threshold),
                    tf.math.maximum(y_pred.sample(self.n_samples), threshold),
                ],
            )

        twcrps = E_1 - self.bias_correction * E_2 / 2

        return twcrps


class MultivariateLoss(tf.keras.losses.Loss):
    """
    Compute losses for multivariate data.

    Facilitates computing losses for multivariate targets
    that may have different units. Allows rescaling the inputs
    and applying weights to each target variables.

    Parameters
    ----------
    metric: {"mse", "mae", "crps_energy"}
        The function used to compute the loss.
    scaling: {"minmax", "standard"}
        (Optional) A scaling to apply to the data, in order to address the differences
        in magnitude due to different units (if unit is not of the variables. Default is `None`.
    weights: array-like
        (Optional) Weights assigned to each variable in the computation of the loss. Default is `None`.
    **kwargs:
        (Optional) Additional keyword arguments to be passed to the parent `Loss` class.


    """

    avail_metrics = {
        "mse": lambda y_true, y_pred: tf.reduce_mean(
            tf.square(y_true - y_pred), axis=0
        ),
        "mae": lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred), axis=0),
        "crps_energy": crps_energy,
    }

    avail_scaling = {
        "standard": "_standard_scaling",
        "minmax": "_minmax_scaling",
        None: "none",
    }

    def __init__(
        self,
        metric: Literal["mse", "mae"],
        scaling: Literal["minmax", "standard"] = None,
        weights: Union[list, np.ndarray] = None,
        **kwargs,
    ) -> None:
        super(MultivariateLoss, self).__init__(**kwargs)

        try:
            self.metric = self.avail_metrics[metric]
        except KeyError as err:
            raise NotImplementedError(
                f"`metric` argument must be one of {list(self.avail_metrics.keys())}"
            ) from err

        try:
            if getattr(self.metric, "loss_type", None) == "probabilistic":
                method_name = self.avail_scaling[scaling] + "_probabilistic"
            else:
                method_name = self.avail_scaling[scaling]
            self.scaling = getattr(self, method_name, None)
        except KeyError as err:
            raise NotImplementedError(
                f"`scaling` argument must be one of {list(self.avail_scaling.keys())}"
            ) from err

        if weights:
            self.weights = tf.constant(weights)
        else:
            self.weights = None

    def get_config(self) -> None:
        config = {
            "metric": self.metric,
            "scaling": self.scaling,
            "weights": self.weights,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: Union[tf.Tensor, np.ndarray, tfp.distributions.Distribution],
    ) -> tf.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        y_true: array-like
            Values representing the ground truth.
        y_pred: array_like or tfp.Distribution
            Predicted values or distributions.

        """

        assert y_true.shape[1:] == y_pred.shape[1:]
        if self.weights is not None:
            assert (
                len(self.weights) == y_true.shape[-1]
            ), "Number weights must match the number of target variables."

        if self.scaling:
            y_true, y_pred = self.scaling(y_true, y_pred)

        loss = self.metric(y_true, y_pred)

        if self.weights is not None:
            loss = tf.multiply(loss, self.weights)

        return loss

    def _minmax_scaling(
        self, y_true: Union[tf.Tensor, np.ndarray], y_pred: Union[tf.Tensor, np.ndarray]
    ) -> tuple[tf.Tensor]:

        y_true_min = tf.reduce_min(y_true, axis=0)
        y_true_max = tf.reduce_max(y_true, axis=0)
        y_true = (y_true - y_true_min) / (y_true_max - y_true_min)
        y_pred = (y_pred - y_true_min) / (y_true_max - y_true_min)

        return y_true, y_pred

    def _minmax_scaling_probabilistic(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: tfp.distributions.Distribution,
    ) -> tuple[tf.Tensor]:

        y_true_min = tf.reduce_min(y_true, axis=0)
        y_true_max = tf.reduce_max(y_true, axis=0)

        scale = tfp.bijectors.Scale(scale=1 / (y_true_max - y_true_min))
        shift = tfp.bijectors.Shift(shift=-y_true_min)
        y_true = (y_true - y_true_min) / (y_true_max - y_true_min)
        y_pred = scale(shift(y_pred))

        y_pred.shape = (*y_pred.batch_shape, *y_pred.event_shape)

        return y_true, y_pred

    def _standard_scaling(
        self, y_true: Union[tf.Tensor, np.ndarray], y_pred: Union[tf.Tensor, np.ndarray]
    ) -> tuple[tf.Tensor]:

        y_true_mean = tf.math.reduce_mean(y_true, axis=0)
        y_true_std = tf.math.reduce_std(y_true, axis=0)
        y_true = (y_true - y_true_mean) / y_true_std
        y_pred = (y_pred - y_true_mean) / y_true_std

        return y_true, y_pred

    def _standard_scaling_probabilistic(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: tfp.distributions.Distribution,
    ) -> tuple[tf.Tensor]:

        y_true_mean = tf.math.reduce_mean(y_true, axis=0)
        y_true_std = tf.math.reduce_std(y_true, axis=0)

        scale = tfp.bijectors.Scale(scale=1 / y_true_std)
        shift = tfp.bijectors.Shift(shift=-y_true_mean)
        y_true = (y_true - y_true_mean) / y_true_std
        y_pred = scale(shift(y_pred))

        y_pred.shape = (*y_pred.batch_shape, *y_pred.event_shape)

        return y_true, y_pred
