import keras
import keras.ops as ops
from torch.distributions import Distribution

from mlpp_lib.probabilistic_layers import WrappingTorchDist


def _expected_value(y_pred):
    if isinstance(y_pred, (Distribution, WrappingTorchDist)):
        return y_pred.mean
    return y_pred


@keras.saving.register_keras_serializable(package="mlpp_lib.metrics")
def expected_bias(y_true, y_pred):
    """Bias of the expected value of the prediction."""
    return ops.mean(_expected_value(y_pred) - y_true, axis=-1)


@keras.saving.register_keras_serializable(package="mlpp_lib.metrics")
def bias(y_true, y_pred):
    """Alias of `expected_bias`, for compatibility with mlpp-lib < 1.0."""
    return expected_bias(y_true, y_pred)


@keras.saving.register_keras_serializable(package="mlpp_lib.metrics")
def expected_mean_absolute_error(y_true, y_pred):
    """Mean absolute error of the expected value of the prediction."""
    return ops.mean(ops.absolute(_expected_value(y_pred) - y_true), axis=-1)


@keras.saving.register_keras_serializable(package="mlpp_lib.metrics")
class MAEBusts(keras.metrics.Metric):
    """Compute frequency of occurrence of absolute errors > threshold."""

    def __init__(self, threshold, name="mae_busts", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.n_busts = self.add_weight(name="nb", initializer="zeros")
        self.n_samples = self.add_weight(name="ns", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.convert_to_tensor(y_true, self.dtype)
        y_pred = _expected_value(y_pred)
        values = ops.cast(ops.abs(y_pred - y_true) > self.threshold, self.dtype)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            # broadcast the weights of each sample to all target dimensions
            while ops.ndim(sample_weight) < ops.ndim(values):
                sample_weight = ops.expand_dims(sample_weight, -1)
            sample_weight = ops.broadcast_to(sample_weight, ops.shape(values))
            values = ops.multiply(values, sample_weight)
            self.n_samples.assign_add(ops.sum(sample_weight))
        else:
            self.n_samples.assign_add(ops.cast(ops.size(values), self.dtype))

        self.n_busts.assign_add(ops.sum(values))

    def result(self):
        return self.n_busts / self.n_samples

    def reset_state(self):
        self.n_busts.assign(0)
        self.n_samples.assign(0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
