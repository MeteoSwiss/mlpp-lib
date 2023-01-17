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
    },
]


@pytest.mark.parametrize("param_run", RUNS)
def test_train(param_run, features_dataset, targets_dataset, splits_train_val):
    results = train.train(
        param_run,
        features_dataset[param_run["features"]],
        targets_dataset[param_run["targets"]],
        splits_train_val,
    )
    assert len(results) == 4
    assert isinstance(results[0], Functional)
    assert isinstance(results[1], dict)
    assert isinstance(results[2], Standardizer)
    assert isinstance(results[3], dict)

    if isinstance(param_run["loss"], str):
        loss_name = param_run["loss"]
    else:
        loss_name = list(param_run["loss"])[0]
    assert loss_name in results[1]