import tensorflow as tf


def bias(y_true, y_pred):
    return tf.reduce_mean(y_pred - y_true, axis=-1)
