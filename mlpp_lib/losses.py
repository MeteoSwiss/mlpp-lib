from typing import Literal, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

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

    use_reparameterization = (
        fct_dist.reparameterization_type == tfd.FULLY_REPARAMETERIZED
    )

    samples_1 = fct_dist.sample(n_samples)
    samples_2 = fct_dist.sample(n_samples)

    # first term
    E_1 = tfp.monte_carlo.expectation(
        f=lambda x: tf.norm(x - obs[None, :], ord=1, axis=-1),
        samples=samples_1,
        log_prob=fct_dist.log_prob,
        use_reparameterization=use_reparameterization,
    )

    # second term
    E_2 = tfp.monte_carlo.expectation(
        f=lambda x: tf.norm(x - samples_2, ord=1, axis=-1),
        samples=samples_1,
        log_prob=fct_dist.log_prob,
        use_reparameterization=use_reparameterization,
    )
    crps = E_1 - E_2 / 2
    # Avoid negative loss when E_2 >> E_1 caused by large values in `sample_2`
    crps = tf.abs(crps)

    return crps[..., None]


class WeightedCRPSEnergy(tf.keras.losses.Loss):
    """
    Compute threshold-weighted CRPS using its kernel score representation.

    Parameters
    ----------
    threshold : float
        The threshold for the weight function within the threshold-weighted CRPS.
        Specifically, the weight function w(x) is 1{x > threshold}.
    n_samples : int, optional
        The number of Monte Carlo samples to be used for estimating expectations. Must be greater than 1.
        Used only if `y_pred` is of type `tfp.Distribution`.
    correct_crps : bool, optional
        If True, applies a bias correction to the CRPS, as detailed in Eq. 4 of Fricker et al. (2013).
    **kwargs : dict, optional
        Additional keyword arguments to pass to the parent `Loss` class.

    Methods
    -------
    call(y_true, y_pred, scale=None):
        Compute the CRPS for predictions and ground truth data. Optionally, compute the
        CRPS over maximum value for blocks of size `scale`.

    Notes
    -----
    - The implemented weight function is w(x) = 1{x > threshold}.
    - For computational stability, a small offset is added to the final CRPS value.
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
        if self.n_samples < 2:
            raise ValueError("n_samples must be > 1")
        self.correct_crps = bool(correct_crps)
        self.bias_correction = n_samples / (n_samples - 1) if correct_crps else 1.0

    def get_config(self) -> dict:
        custom_config = {
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "correct_crps": self.correct_crps,
        }
        config = super().get_config()
        config.update(custom_config)
        return config

    def call(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: Union[tf.Tensor, np.ndarray, tfp.distributions.Distribution],
        scale: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        y_true: array-like
            Values representing the ground truth.
        y_pred: array_like or tfp.Distribution
            Predicted values or distributions.
        scale : int, optional
            If provided, the CRPS is computed over the maximum values within blocks of
            size `scale` in both `y_true` and `y_pred`. The input tensors' first dimension
            (typically batch size) should be divisible by this scale.
        """

        threshold = tf.constant(self.threshold, dtype=y_true.dtype)
        n_samples = self.n_samples
        y_true = tf.debugging.check_numerics(y_true, "Target values")
        v_obs = tf.math.maximum(y_true, threshold)

        if tf.is_tensor(y_pred) or isinstance(y_pred, np.ndarray):

            n_samples = y_pred.shape[0]

            v_ens = tf.math.maximum(y_pred, threshold)

            if scale is not None and scale > 1:
                # Reshape the tensors and compute the block maxima
                reshaped_true = tf.reshape(v_obs, (-1, scale, 1))
                v_obs = tf.reduce_max(reshaped_true, axis=1)

                reshaped_pred = tf.reshape(v_ens, (n_samples, -1, scale, 1))
                v_ens = tf.reduce_max(reshaped_pred, axis=2)

            # first term
            E_1 = tf.abs(v_ens - v_obs[None, :])
            E_1 = tf.reduce_mean(E_1, axis=0)

            # second term
            E_2 = tf.abs(v_ens[None, :] - v_ens[:, None])
            E_2 = tf.reduce_mean(E_2, axis=(0, 1))

        else:

            use_reparameterization = (
                y_pred.reparameterization_type == tfd.FULLY_REPARAMETERIZED
            )

            samples_1 = tf.math.maximum(y_pred.sample(n_samples), threshold)
            samples_2 = tf.math.maximum(y_pred.sample(n_samples), threshold)

            if scale is not None and scale > 1:
                # Reshape the tensors and compute the block maxima
                reshaped_true = tf.reshape(v_obs, (-1, scale, 1))
                v_obs = tf.reduce_max(reshaped_true, axis=1)

                reshaped_pred = tf.reshape(samples_1, (n_samples, -1, scale, 1))
                samples_1 = tf.reduce_max(reshaped_pred, axis=2)

                reshaped_pred = tf.reshape(samples_2, (n_samples, -1, scale, 1))
                samples_2 = tf.reduce_max(reshaped_pred, axis=2)

            # first term
            E_1 = tfp.monte_carlo.expectation(
                f=lambda x: tf.norm(x - v_obs[None, :], ord=1, axis=-1),
                samples=samples_1,
                log_prob=y_pred.log_prob,
                use_reparameterization=use_reparameterization,
            )[..., None]

            # second term
            E_2 = tfp.monte_carlo.expectation(
                f=lambda x: tf.norm(x - samples_2, ord=1, axis=-1),
                samples=samples_1,
                log_prob=y_pred.log_prob,
                use_reparameterization=use_reparameterization,
            )[..., None]

        twcrps = E_1 - self.bias_correction * E_2 / 2

        # Avoid negative loss when E_2 >> E_1 caused by large values in `sample_2`
        twcrps = tf.abs(twcrps)
        # Add a small offset to ensure stability
        twcrps += 1e-7

        return twcrps


class MultiScaleCRPSEnergy(WeightedCRPSEnergy):
    """
    Compute threshold-weighted CRPS over multiple scales of data.

    Parameters
    ----------
    scales: list[int]
        List of scales (block sizes) over which CRPS is computed. The batch size used for
        training must be divisible by all scales.
    threshold: float
        The threshold to be used within the weight function of the threshold-weighted CRPS.
    n_samples: int, optional (default=1000)
        Number of samples used to compute the Monte Carlo expectations.
    correct_crps: bool, optional (default=True)
        Whether to bias correct the CRPS following Eq. 4 in Fricker et al. (2013).
    **kwargs:
        (Optional) Additional keyword arguments to be passed to the parent `WeightedCRPSEnergy` class.

    Notes
    -----
    The CRPS is computed over the maximum value in each block (scale) of `y_true`
    and the samples sampled from `y_pred`.
    """

    def __init__(self, scales: list[int], **kwargs) -> None:
        """Initialize the MultiScaleCRPSEnergy with scales and other parameters."""
        super(MultiScaleCRPSEnergy, self).__init__(**kwargs)
        self.scales = scales

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: Union[tf.Tensor, tfp.distributions.Distribution],
    ) -> tf.Tensor:
        """
        Compute the threshold-weighted CRPS over multiple scales of data.

        Parameters
        ----------
        y_true: tf.Tensor
            Ground truth tensor.
        y_pred: Union[tf.Tensor, tfp.distributions.Distribution]
            Predicted tensor or distribution.

        Returns
        -------
        tf.Tensor
            Loss tensor, computed as the average CRPS over the provided scales.
        """

        all_losses = []

        for scale in self.scales:

            if scale > 1:
                tf.debugging.assert_equal(
                    tf.shape(y_true)[0] % scale,
                    0,
                    message=f"Input tensor length ({tf.shape(y_true)[0]}) is not divisible by scale {scale}.",
                )

            scale_loss = super(MultiScaleCRPSEnergy, self).call(
                y_true, y_pred, scale=scale
            )

            # Repeat loss to match the original shape for CRPS computation
            if scale > 1:
                scale_loss = tf.repeat(scale_loss, scale, axis=0)

            all_losses.append(scale_loss)

        # Average the losses over all scales
        total_loss = tf.reduce_mean(tf.stack(all_losses, axis=0), axis=0)

        return total_loss


class EnergyScore(tf.keras.losses.Loss):
    """
    Compute Energy Score.

    Parameters
    ----------
    threshold: float
        The threshold to be used within the weight function of the Energy Score.
    n_samples: int
        Number of samples used to compute the Monte Carlo expectations.
    **kwargs:
        (Optional) Additional keyword arguments to be passed to the parent `Loss` class.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        **kwargs,
    ) -> None:
        super(EnergyScore, self).__init__(**kwargs)

        self.n_samples = int(n_samples)

    def get_config(self) -> None:
        custom_config = {
            "n_samples": self.n_samples,
        }
        config = super().get_config()
        config.update(custom_config)
        return config

    def call(
        self,
        y_true: Union[tf.Tensor, np.ndarray],
        y_pred: tfp.distributions.Distribution,
    ) -> tf.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        y_true: array-like
            Values representing the ground truth.
        y_pred: tfp.Distribution
            Predicted distributions.
        """

        y_true = tf.debugging.check_numerics(y_true, "Target values")

        use_reparameterization = (
            y_pred.reparameterization_type == tfd.FULLY_REPARAMETERIZED
        )

        samples_1 = y_pred.sample(self.n_samples)
        samples_2 = y_pred.sample(self.n_samples)

        # first term
        E_1 = tfp.monte_carlo.expectation(
            f=lambda x: tf.norm(x - y_true[None, ...], ord=1, axis=-1),
            samples=samples_1,
            log_prob=y_pred.log_prob,
            use_reparameterization=use_reparameterization,
        )

        E_2 = tfp.monte_carlo.expectation(
            f=lambda x: tf.norm(x - samples_2, ord=1, axis=-1),
            samples=samples_1,
            log_prob=y_pred.log_prob,
            use_reparameterization=use_reparameterization,
        )

        energy_score = E_1 - E_2 / 2

        # Avoid negative loss when E_2 >> E_1 caused by large values in `sample_2`
        energy_score = tf.abs(energy_score)

        return energy_score


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


class BinaryClassifierLoss(tf.keras.losses.Loss):
    """
    Compute binary classification loss from continuous predictions based on a threshold.

    Parameters
    ----------
    threshold: float
    loss_type: {"binary_crossentropy", "focal"}
        The type of loss to be used.
    n_samples: int, optional
    **kwargs:
        (Optional) Additional keyword arguments to be passed to the parent `Loss` class.

    """

    def __init__(
        self,
        threshold: float,
        loss_type: Literal["binary_crossentropy", "focal"] = "binary_crossentropy",
        n_samples: int = 1000,
        **kwargs,
    ) -> None:
        super(BinaryClassifierLoss, self).__init__(**kwargs)

        self.threshold = float(threshold)
        self.n_samples = int(n_samples)
        if self.n_samples < 2:
            raise ValueError("n_samples must be > 1")
        self.loss_type = loss_type

    def get_config(self) -> dict:
        custom_config = {
            "threshold": self.threshold,
            "loss_type": self.loss_type,
            "n_samples": self.n_samples,
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
        n_samples = self.n_samples
        y_true = tf.debugging.check_numerics(y_true, "Target values")

        if isinstance(y_pred, tfp.distributions.Distribution):
            y_pred_samples = y_pred.sample(n_samples)
        else:
            y_pred_samples = y_pred

        y_pred_samples = tf.debugging.check_numerics(y_pred_samples, "Predicted values")

        y_true_bool = tf.cast(y_true > threshold, dtype=y_true.dtype)
        y_pred_bool = tf.cast(y_pred_samples > threshold, dtype=y_true.dtype)
        y_pred_prob = tf.reduce_mean(y_pred_bool, axis=0)

        loss = tf.keras.losses.binary_crossentropy(y_true_bool, y_pred_prob, axis=1)
        if self.loss_type == "focal":
            loss = tf.pow(1 - tf.exp(-loss), 2)

        return loss


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, losses):
        # Local import to avoid circular dependency with mlpp_lib.utils
        from mlpp_lib.utils import get_loss

        super(CombinedLoss, self).__init__()
        self.losses = []
        self.weights = []

        # Initialize losses based on the input config dictionaries
        for loss_config in losses:
            self.weights.append(loss_config.get("weight", 1.0))
            self.losses.append(get_loss(loss_config))

    def call(self, y_true, y_pred):
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(y_true, y_pred)
        return total_loss
