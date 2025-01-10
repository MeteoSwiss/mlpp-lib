import keras
import keras.ops as ops

def bias(y_true, y_pred):
    return ops.mean(y_pred - y_true, axis=-1)


class MAEBusts(keras.metrics.Metric):
    """Compute frequency of occurrence of absolute errors > threshold."""

    def __init__(self, threshold, name="mae_busts", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.n_busts = self.add_weight(name="nb", initializer="zeros")
        self.n_samples = self.add_weight(name="ns", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = ops.cast(ops.abs(y_pred - y_true) > self.threshold, self.dtype)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
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
