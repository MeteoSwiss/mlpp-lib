import json

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
            "ProperScores": {"thresholds": [0, 1, 2]},
        },
    },
]


@pytest.mark.parametrize("param_run", RUNS)
def test_train(
    param_run, features_dataset, targets_dataset, splits_train_val, tmp_path
):
    param_run.update({"epochs": 3})
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

    # try to dump fit history to json
    with open(tmp_path / "history.json", "w") as f:
        json.dump(results[3], f, indent=2)

    if isinstance(param_run["loss"], str):
        loss_name = param_run["loss"]
    else:
        loss_name = list(param_run["loss"])[0]
    assert loss_name in results[1]
