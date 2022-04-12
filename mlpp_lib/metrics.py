import tensorflow as tf


def bias(y_true, y_pred) -> tf.Tensor:
    difference = y_pred - y_true
    return tf.reduce_mean(difference, axis=-1)
