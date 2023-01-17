import logging
from pprint import pformat
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr

from mlpp_lib.callbacks import TimeHistory, ProperScores
from mlpp_lib.datasets import get_tensor_dataset, split_dataset
from mlpp_lib.standardizers import standardize_split_dataset
from mlpp_lib.utils import (
    get_callback,
    get_loss,
    get_metric,
    get_model,
    process_out_bias_init,
)


LOGGER = logging.getLogger(__name__)


def get_log_params(param_run: dict) -> dict:
    """Extract a selection of parameters for pretty logging"""
    log_params = {}
    for dimension in param_run["data_partitioning"].keys():
        params = {
            f"partitioning_{dimension}_{name}": value
            for name, value in param_run["data_partitioning"][dimension].items()
        }
        log_params.update(params)
    # log_params["features_names"] = param_run["features"]
    # Note to future self: the list of features can easily exceed the maximum length for
    # a logged parameter, so we excluded it. Instead, this is logged as an artifact
    # together with the input run parameters.
    log_params["targets_names"] = param_run["targets"]
    log_params["sample_weights_names"] = param_run.get("sample_weights")
    log_params["event_dims"] = param_run["batching"]["event_dims"]
    log_params["thinning"] = param_run.get("thinning")
    log_params.update(param_run["filter"])
    log_params["model_name"] = list(param_run["model"])[0]
    log_params.update(param_run["model"][log_params["model_name"]])
    log_params["loss"] = param_run["loss"]
    return log_params


def train(
    param_run: dict,
    features: xr.Dataset,
    targets: xr.Dataset,
    splits_train_val: dict,
    sample_weights: Optional[xr.Dataset] = None,
    targets_mask: Optional[xr.DataArray] = None,
    features_mask: Optional[xr.DataArray] = None,
    callbacks: Optional[list] = None,
) -> tuple:

    LOGGER.debug(f"run params:\n{pformat(param_run)}")
    # required parameters
    model_config = param_run["model"]
    loss_config = param_run["loss"]
    # optional parameters
    metrics_config = param_run.get("metrics", [])
    callbacks_config = param_run.get("callbacks", {})
    event_dims = param_run.get("batching", {}).get("event_dims", [])
    batch_size = param_run.get("batching", {}).get("batch_size")
    shuffle = param_run.get("batching", {}).get("shuffle", True)
    out_bias_init = param_run.get("out_bias_init")
    thinning = param_run.get("thinning")
    optimizer = param_run.get("optimizer", "Adam")
    learning_rate = param_run.get("learning_rate", 0.001)
    epochs = param_run.get("epochs", 1)
    steps_per_epoch = param_run.get("steps_per_epoch")

    # load data and filter measurements
    if targets_mask is not None:
        targets = targets.where(
            targets_mask
        )  # TODO: make sure that qa code follows this!
        LOGGER.info(
            f"Will keep {targets_mask.sum()/targets_mask.size * 100:.1f}% of targets."
        )

    if features_mask is not None:
        LOGGER.info(
            f"Will keep {features_mask.sum() / features_mask.size * 100:.1f}% of features."
        )
        features = features.assign_coords(is_valid=features_mask)
    else:
        features = features.assign_coords(is_valid=True)

    # split datasets
    features = split_dataset(features, splits=splits_train_val, thinning=thinning)
    targets = split_dataset(targets, splits=splits_train_val, thinning=thinning)
    sample_weights = split_dataset(
        sample_weights, splits=splits_train_val, thinning=thinning
    )

    # standardize features
    features, standardizer = standardize_split_dataset(
        features, return_standardizer=True
    )

    # reshape as input tensors
    data = {}
    for split_key in splits_train_val.keys():
        features[split_key] = features[split_key].where(features[split_key].is_valid)
        data[split_key] = get_tensor_dataset(
            features[split_key],
            targets[split_key],
            sample_weights[split_key],
            event_dims=event_dims,
        )
        LOGGER.info(
            f"{data[split_key][0].sizes['sample']} samples in the {split_key} set."
        )
        if data[split_key][2] is not None:
            # take the product in case multiple weights are used
            var_axis = data[split_key][2].dims.index("variable")
            data[split_key][2] = np.prod(data[split_key][2], axis=var_axis)

    x_train_data = data["train"][0].values
    y_train_data = data["train"][1].values
    x_val_data = data["val"][0].values
    y_val_data = data["val"][1].values
    w_train_data = data["train"][2]
    # see https://github.com/keras-team/keras/pull/16177
    if w_train_data is not None:
        w_train_data = pd.Series(w_train_data)
    del data

    # prepare model
    out_bias_init = process_out_bias_init(y_train_data, out_bias_init, event_dims)
    model_config[list(model_config)[0]].update({"out_bias_init": out_bias_init})
    input_shape = x_train_data.shape[1:]
    output_shape = y_train_data.shape[1:]
    model = get_model(input_shape, output_shape, model_config)
    loss = get_loss(loss_config)
    metrics = [get_metric(metric) for metric in metrics_config]
    optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(print_fn=LOGGER.info)

    # callbacks
    if callbacks is None:
        callbacks = []

    for callback in callbacks_config.items():
        callback_instance = get_callback({callback[0]: callback[1]})

        if isinstance(callback_instance, ProperScores):
            callback_instance.add_validation_data((x_val_data, y_val_data))

        callbacks.append(callback_instance)

    time_callback = TimeHistory()
    callbacks.append(time_callback)

    LOGGER.info("Start training.")
    res = model.fit(
        x=x_train_data,
        y=y_train_data,
        sample_weight=w_train_data,
        epochs=epochs,
        validation_data=(x_val_data, y_val_data),
        callbacks=callbacks,
        shuffle=shuffle,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
    )
    LOGGER.info("Done! \U0001F40D")

    custom_objects = tf.keras.layers.serialize(model)
    if isinstance(loss_config, dict):
        loss_name = list(loss_config)[0]
    else:
        loss_name = loss_config
    custom_objects[loss_name] = loss
    history = res.history

    # for some reasons, 'lr' is provided as float32
    # and needs to be casted in order to be serialized
    if "lr" in history:
        history["lr"] = list(map(float, history["lr"]))

    return model, custom_objects, standardizer, history
