import tensorflow as tf


def bias(y_true, y_pred):
    return tf.reduce_mean(y_pred - y_true, axis=-1)


class MAEBusts(tf.keras.metrics.Metric):
    """Compute frequency of occurrence of absolute errors > thr"""

    def __init__(self, threshold, name="mae_busts", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.n_busts = self.add_weight(name="nb", initializer="zeros")
        self.n_samples = self.add_weight(name="ns", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.abs(y_pred - y_true) > self.threshold, tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)
            self.n_samples.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.n_samples.assign_add(tf.cast(tf.size(values), tf.float32))

        self.n_busts.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.n_busts / self.n_samples

    def reset_state(self):
        self.n_busts.assign(0)
        self.n_samples.assign(0)
