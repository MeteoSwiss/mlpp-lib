import json

import cloudpickle
import pytest
from keras.engine.functional import Functional

from mlpp_lib import train
from mlpp_lib.standardizers import Standardizer


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
        "callbacks": {"EarlyStopping": {"patience": 10, "restore_best_weights": True}},
    },
    # use a more complicated loss function
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "model": {
            "fully_connected_network": {
                "hidden_layers": [10],
                "probabilistic_layer": "IndependentNormal",
            }
        },
        "loss": {"WeightedCRPSEnergy": {"threshold": 0, "n_samples": 5}},
        "metrics": ["bias", "mean_absolute_error", {"MAEBusts": {"threshold": 0.5}}],
    },
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
        "metrics": ["bias"],
        "callbacks": {
            "EarlyStopping": {
                "patience": 10,
                "restore_best_weights": True,
                "verbose": 1,
            },
            "ReduceLROnPlateau": {"patience": 1, "verbose": 1},
            "EnsembleMetrics": {"thresholds": [0, 1, 2]},
        },
    },
]


@pytest.mark.parametrize("param_run", RUNS)
def test_train(param_run, features_dataset, targets_dataset, splits_train_val):
    num_epochs = 2
    param_run.update({"epochs": num_epochs})
    results = train.train(
        param_run,
        features_dataset[param_run["features"]],
        targets_dataset[param_run["targets"]],
        splits_train_val,
    )
    assert len(results) == 4
    assert isinstance(results[0], Functional)  # model
    assert isinstance(results[1], dict)  # custom_objects
    assert isinstance(results[2], Standardizer)  # standardizer
    assert isinstance(results[3], dict)  # history

    assert all([len(v) == num_epochs for v in results[3].values()])

    # try to pickle the custom objects
    cloudpickle.dumps(results[1])

    # try to dump fit history to json
    json.dumps(results[3])
