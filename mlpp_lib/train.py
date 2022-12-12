import logging
from pprint import pformat
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from mlpp_lib.callbacks import TimeHistory
from mlpp_lib.datasets import get_tensor_dataset, split_dataset
from mlpp_lib.standardizers import standardize_split_dataset
from mlpp_lib.utils import get_loss, get_model, process_out_bias_init


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
    log_params["sample_weights"] = param_run.get("sample_weights")
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

    # run parameters
    LOGGER.debug(f"run params:\n{pformat(param_run)}")
    event_dims = param_run["batching"]["event_dims"]
    batch_size = param_run["batching"]["batch_size"]
    shuffle = param_run["batching"]["shuffle"]
    patience = param_run.get("patience")
    thinning = param_run.get("thinning")

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

    # prepare model
    out_bias_init = param_run.get("out_bias_init")
    out_bias_init = process_out_bias_init(data["train"][1], out_bias_init, event_dims)
    param_run["out_bias_init"] = out_bias_init
    input_shape = data["train"][0].shape[1:]
    output_shape = data["train"][1].shape[1:]
    model = get_model(input_shape, output_shape, param_run["model"])
    loss = get_loss(param_run["loss"])
    optimizer = getattr(tf.keras.optimizers, param_run["optimizer"])(
        learning_rate=param_run["learning_rate"]
    )
    model.compile(optimizer=optimizer, loss=loss)
    model.summary(print_fn=LOGGER.info)

    # prepare training
    if callbacks is None:
        callbacks = []
    if patience:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=patience,
                monitor="val_loss",
                restore_best_weights=True,
            )
        )

    time_callback = TimeHistory()
    callbacks.append(time_callback)

    x_train_data = data["train"][0].values
    y_train_data = data["train"][1].values
    x_val_data = data["val"][0].values
    y_val_data = data["val"][1].values
    w_train_data = data["train"][2]

    # see https://github.com/keras-team/keras/pull/16177
    if w_train_data is not None:
        w_train_data = pd.Series(w_train_data)

    LOGGER.info("Start training.")
    history = model.fit(
        x=x_train_data,
        y=y_train_data,
        sample_weight=w_train_data,
        epochs=param_run["epochs"],
        validation_data=(x_val_data, y_val_data),
        callbacks=callbacks,
        shuffle=shuffle,
        batch_size=batch_size,
        verbose=1,
    )
    LOGGER.info("Done! \U0001F40D")

    custom_objects = tf.keras.layers.serialize(model)
    if isinstance(param_run["loss"], dict):
        loss_name = list(param_run["loss"].keys())[0]
    else:
        loss_name = param_run["loss"]
    custom_objects[loss_name] = loss
    history = history.history
    history["time"] = time_callback.times

    return model, custom_objects, standardizer, history
