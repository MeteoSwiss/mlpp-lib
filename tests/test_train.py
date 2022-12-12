import pytest
from keras.engine.functional import Functional

from mlpp_lib import train
from mlpp_lib.standardizers import Standardizer

RUNS = [
    # minimal set of parameters
    {
        "features": ["coe:x1"],
        "targets": ["obs:y1"],
        "batching": {
            "event_dims": [],
            "batch_size": 1000,
            "shuffle": True,
        },
        "model": {
            "fully_connected_network": {
                "hidden_layers": [3, 2, 1],
                "activations": "relu",
                "dropout": [0.5, 0.0, 0.0],
                "probabilistic_layer": "IndependentGamma",
            }
        },
        "loss": {"WeightedCRPSEnergy": {"threshold": 0}},
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "epochs": 1,
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
