import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import xarray as xr
from tensorflow import keras


LEADTIMES = np.arange(24)
REFTIMES = pd.date_range("2018-01-01", "2018-03-31", freq="24H")
STATIONS = [chr(i) * 3 for i in range(ord("A"), ord("Z"))]
SHAPE = (len(REFTIMES), len(LEADTIMES), len(STATIONS))
DIMS = ["forecast_reference_time", "t", "station"]


@pytest.fixture
def features_dataset() -> xr.Dataset:
    """
    Create a dataset as if it was loaded from `features.zarr`.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal(size=(*SHAPE, 4))
    X[(X > 4.5) | (X < -4.5)] = np.nan

    features = xr.Dataset(
        {
            "coe:x1": (DIMS, X[..., 0]),
            "coe:x2": (DIMS, X[..., 1]),
            "obs:x3": (DIMS, X[..., 2]),
            "dem:x4": (DIMS, X[..., 3]),
        },
        coords={
            "forecast_reference_time": REFTIMES,
            "t": LEADTIMES,
            "station": STATIONS,
        },
    )

    return features


@pytest.fixture
def targets_dataset() -> xr.Dataset:
    """
    Create a dataset as if it was loaded from `targets.zarr`.
    """
    rng = np.random.default_rng(1)
    Y = rng.standard_normal(size=(*SHAPE, 2))
    Y[(Y > 4.5) | (Y < -4.5)] = np.nan

    targets = xr.Dataset(
        {"obs:y1": (DIMS, Y[..., 0]), "obs:y2": (DIMS, Y[..., 1])},
        coords={
            "forecast_reference_time": REFTIMES,
            "t": LEADTIMES,
            "station": STATIONS,
        },
    )

    return targets


@pytest.fixture
def get_dummy_keras_model() -> tf.keras.Model:
    def _model(n_inputs, n_outpus):
        inputs = tf.keras.Input(shape=(n_inputs,))
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(n_outpus, activation=tf.nn.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
            loss=keras.losses.CategoricalCrossentropy(),
        )

        return model

    return _model
