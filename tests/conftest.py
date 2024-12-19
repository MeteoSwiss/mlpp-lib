import numpy as np
import pandas as pd
import pytest
import xarray as xr
import keras

from mlpp_lib.utils import get_loss, get_model


LEADTIMES = np.arange(24)
REFTIMES = pd.date_range("2018-01-01", "2018-03-31", freq="24h")
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
def features_multi() -> xr.Dataset:
    """
    Create a dataset as if it was loaded from `features.zarr`.
    Coherent with the number of data transformations defined in the normalizers module.
    """
    import mlpp_lib.normalizers as no

    rng = np.random.default_rng(1)
    X = rng.standard_normal(
        size=(*SHAPE, len([n.name for n in no.DataTransformation.__subclasses__()]))
    )
    X = np.float64(X)
    X[(X > 4.5) | (X < -4.5)] = np.nan
    features = xr.Dataset(
        {f"var{i}": (DIMS, X[..., i]) for i in range(X.shape[-1])},
        coords={
            "forecast_reference_time": REFTIMES,
            "t": LEADTIMES,
            "station": STATIONS,
        },
    )

    return features


@pytest.fixture
def datatransformations() -> list:
    """
    Create a list of data transformations.
    The list consists of all available data transformations.
    """
    import mlpp_lib.normalizers as no

    datatransformations = [
        no.create_transformation_from_str(
            n.name, inputs={"fillvalue": -5} if n.name == "Identity" else {}
        )  # temporary fix, do we want to let the user define different fillvalue for each transformation ?
        for n in no.DataTransformation.__subclasses__()
    ]

    return datatransformations


@pytest.fixture
def data_transformer() -> xr.Dataset:
    """
    Create a datatransformer.
    """
    import mlpp_lib.normalizers as no

    transformations_list = [n.name for n in no.DataTransformation.__subclasses__()]
    method_var_dict = {
        transformation: [f"var{i}"]
        for i, transformation in enumerate(transformations_list)
    }
    data_transformer = no.DataTransformer(method_var_dict)
    data_transformer.transformers["Identity"][
        0
    ].fillvalue = (
        -5
    )  # temporary fix, do we want to let the user define different fillvalue for each transformation ?

    return data_transformer


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
def splits_train_val() -> dict:
    """ """
    splits = {
        "train": {
            "forecast_reference_time": REFTIMES[: len(REFTIMES) // 2],
            "station": STATIONS[: len(STATIONS) // 2],
        },
        "val": {
            "forecast_reference_time": REFTIMES[len(REFTIMES) // 2 :],
            "station": STATIONS[len(STATIONS) // 2 :],
        },
    }

    return splits


@pytest.fixture
def get_dummy_keras_model() -> keras.Model:
    def _model(n_inputs, n_outpus):
        inputs = keras.Input(shape=(n_inputs,))
        x = keras.layers.Dense(4, activation=keras.ops.relu)(inputs)
        outputs = keras.layers.Dense(n_outpus, activation=keras.ops.softmax)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
            loss=keras.losses.CategoricalCrossentropy(),
        )

        return model

    return _model


@pytest.fixture
def get_prob_model() -> keras.Model:
    def _model(n_inputs, n_outputs):
        model_config = {
            "fully_connected_network": {
                "hidden_layers": [1],
                "probabilistic_layer": "IndependentNormal",
            }
        }
        model = get_model(n_inputs, n_outputs, model_config)
        loss = get_loss({'DistributionLossWrapper': 'scoringrules.crps_normal'})
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
            loss=loss,
        )
        return model

    return _model
