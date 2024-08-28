import json

import cloudpickle
import numpy as np
import pytest
from keras.engine.functional import Functional
import xarray as xr

from mlpp_lib import train
from mlpp_lib.normalizers import DataTransformer
from mlpp_lib.datasets import DataModule, DataSplitter

from .test_model_selection import ValidDataSplitterOptions


RUNS = [
    # minimal set of parameters
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "IndependentNormal",
            }
        },
        "loss": "crps_energy",
        "optimizer": "RMSprop",
        "callbacks": [
            {"EarlyStopping": {"patience": 10, "restore_best_weights": True}}
        ],
    },
    # use a more complicated loss function
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "IndependentBeta",
            }
        },
        "loss": {"WeightedCRPSEnergy": {"threshold": 0, "n_samples": 5}},
        "optimizer": {"Adam": {"learning_rate": 0.1, "beta_1": 0.95}},
        "metrics": ["bias", "mean_absolute_error", {"MAEBusts": {"threshold": 0.5}}],
    },
    #
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "IndependentNormal",
                "skip_connection": True,
            }
        },
        "loss": "crps_energy",
        "metrics": ["bias"],
        "callbacks": [
            {
                "EarlyStopping": {
                    "patience": 10,
                    "restore_best_weights": True,
                    "verbose": 1,
                }
            },
            {"ReduceLROnPlateau": {"patience": 1, "verbose": 1}},
            {"EnsembleMetrics": {"thresholds": [0, 1, 2]}},
        ],
    },
    # with multiscale CRPS loss
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "IndependentNormal",
            }
        },
        "group_samples": {"t": 2},
        "loss": {
            "MultiScaleCRPSEnergy": {"scales": [1, 2], "threshold": 0, "n_samples": 5}
        },
        "metrics": ["bias"],
    },
    # with combined loss
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "IndependentNormal",
            }
        },
        "loss": {
            "CombinedLoss": {
                "losses": [
                    {"BinaryClassifierLoss": {"threshold": 1}, "weight": 0.7},
                    {"WeightedCRPSEnergy": {"threshold": 0.1}, "weight": 0.1},
                ],
            }
        },
    },
]


@pytest.fixture  # https://docs.pytest.org/en/6.2.x/tmpdir.html
def write_datasets_zarr(tmp_path, features_dataset, targets_dataset):
    features_dataset.to_zarr(tmp_path / "features.zarr", mode="w")
    targets_dataset.to_zarr(tmp_path / "targets.zarr", mode="w")


@pytest.mark.skipif("zarr" not in xr.backends.list_engines(), reason="missing zarr")
@pytest.mark.usefixtures("write_datasets_zarr")
@pytest.mark.parametrize("cfg", RUNS)
def test_train_fromfile(tmp_path, cfg):
    num_epochs = 3
    cfg.update({"epochs": num_epochs})

    splitter_options = ValidDataSplitterOptions(time="lists", station="lists")
    splitter = DataSplitter(splitter_options.time_split, splitter_options.station_split)
    batch_dims = ["forecast_reference_time", "t", "station"]
    datamodule = DataModule(
        features=cfg["features"],
        targets=cfg["targets"],
        batch_dims=batch_dims,
        splitter=splitter,
        data_dir=tmp_path.as_posix() + "/",
    )
    results = train.train(cfg, datamodule)

    assert len(results) == 4
    assert isinstance(results[0], Functional)  # model
    assert isinstance(results[1], dict)  # custom_objects
    assert isinstance(results[2], DataTransformer)  # standardizer
    assert isinstance(results[3], dict)  # history

    assert all([np.isfinite(v).all() for v in results[3].values()])
    assert all([len(v) == num_epochs for v in results[3].values()])

    # try to pickle the custom objects
    cloudpickle.dumps(results[1])

    # try to dump fit history to json
    json.dumps(results[3])


@pytest.mark.parametrize("cfg", RUNS)
def test_train_fromds(features_dataset, targets_dataset, cfg):
    num_epochs = 3
    cfg.update({"epochs": num_epochs})

    splitter_options = ValidDataSplitterOptions(time="lists", station="lists")
    splitter = DataSplitter(splitter_options.time_split, splitter_options.station_split)
    batch_dims = ["forecast_reference_time", "t", "station"]
    datamodule = DataModule(
        features_dataset[cfg["features"]],
        targets_dataset[cfg["targets"]],
        batch_dims,
        splitter,
        group_samples=cfg.get("group_samples"),
    )
    results = train.train(cfg, datamodule)

    assert len(results) == 4
    assert isinstance(results[0], Functional)  # model
    assert isinstance(results[1], dict)  # custom_objects
    assert isinstance(results[2], DataTransformer)  # standardizer
    assert isinstance(results[3], dict)  # history

    assert all([np.isfinite(v).all() for v in results[3].values()])
    assert all([len(v) == num_epochs for v in results[3].values()])

    # try to pickle the custom objects
    cloudpickle.dumps(results[1])

    # try to dump fit history to json
    json.dumps(results[3])
