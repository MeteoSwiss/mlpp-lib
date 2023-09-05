import subprocess
from inspect import getmembers, isfunction, isclass

import numpy as np
import pytest
import tensorflow as tf
from keras.engine.functional import Functional

from mlpp_lib import models
from mlpp_lib import losses, metrics
from mlpp_lib import probabilistic_layers
from mlpp_lib.utils import get_loss, get_metric


def _belongs_here(obj, module):
    return obj[1].__module__ == module.__name__


ALL_LAYERS = [
    obj[0]
    for obj in getmembers(probabilistic_layers, isclass)
    if _belongs_here(obj, probabilistic_layers)
]

ALL_LOSSES = [
    obj[0]
    for obj in getmembers(losses, isfunction) + getmembers(losses, isclass)
    if _belongs_here(obj, losses)
]

ALL_METRICS = [
    obj[0]
    for obj in getmembers(metrics, isfunction) + getmembers(metrics, isclass)
    if _belongs_here(obj, metrics)
]


TEST_LOSSES = [
    "crps_energy_ensemble",
    "crps_energy",
    {"WeightedCRPSEnergy": {"threshold": 0}},
    {"MultivariateLoss": {"metric": "mse"}},
]

TEST_METRICS = [
    "bias",
    "mean_absolute_error",
    {"MAEBusts": {"threshold": 0.5}},
]


@pytest.mark.parametrize("save_format", ["tf", "h5"])
@pytest.mark.parametrize("loss", TEST_LOSSES)
@pytest.mark.parametrize("prob_layer", ALL_LAYERS)
def test_save_model(save_format, loss, prob_layer, tmp_path):
    """Test model save/load"""

    if save_format == "h5":
        tmp_path = f"{tmp_path}.h5"
        save_traces = True  # default value
    else:
        tmp_path = f"{tmp_path}"
        save_traces = False

    model = models.fully_connected_network(
        (5,), 2, hidden_layers=[3], probabilistic_layer=prob_layer
    )
    assert isinstance(model.from_config(model.get_config()), Functional)
    loss = get_loss(loss)
    metrics = [get_metric(metric) for metric in TEST_METRICS]
    model.compile(loss=loss, metrics=metrics)
    model.save(tmp_path, save_traces=save_traces)

    # test trying to load the model from a new process
    # this is a bit slow, since each process needs to reload all the dependencies ...

    # not compiling
    args = [
        "python",
        "-c",
        "import tensorflow as tf;"
        f"from mlpp_lib.probabilistic_layers import {prob_layer};"
        f"tf.keras.saving.load_model('{tmp_path}', compile=False)",
    ]
    completed_process = subprocess.run(args, shell=True)
    assert completed_process.returncode == 0, "failed to reload model"

    # compiling
    args = [
        "python",
        "-c",
        "import tensorflow as tf;"
        f"from mlpp_lib.losses import {loss};"
        f"from mlpp_lib.probabilistic_layers import {prob_layer};"
        f"tf.keras.saving.load_model('{tmp_path}', custom_objects={{'{loss}':{loss}}})",
    ]
    completed_process = subprocess.run(args, shell=True)
    assert completed_process.returncode == 0, "failed to reload model"

    input_arr = tf.random.uniform((1, 5))
    outputs = model(input_arr).mean()
    del model
    tf.keras.backend.clear_session()
    model = tf.keras.saving.load_model(tmp_path, compile=False)
    assert isinstance(model, Functional)
    np.testing.assert_allclose(model(input_arr).mean(), outputs)
