import logging
from pprint import pformat
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from mlpp_lib.callbacks import TimeHistory
from mlpp_lib.datasets import get_tensor_dataset, split_dataset, DataModule, Dataset
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
    cfg: dict,
    datamodule: DataModule,
    callbacks: Optional[list] = None,
) -> tuple:

    LOGGER.debug(f"run params:\n{pformat(cfg)}")

    datamodule.setup("fit")
    model_config = cfg["model"]
    loss_config = cfg["loss"]

    # prepare model
    out_bias_init = process_out_bias_init(
        datamodule.train.x, cfg.get("out_bias_init", "zeros"), cfg["event_dims"]
    )
    model_config[list(model_config)[0]].update({"out_bias_init": out_bias_init})
    input_shape = datamodule.train.x.shape[1:]
    output_shape = datamodule.train.y.shape[1:]
    model: tf.keras.Model = get_model(input_shape, output_shape, model_config)
    loss = get_loss(loss_config)
    optimizer = getattr(tf.keras.optimizers, optimizer)(
        learning_rate=cfg.get("learning_rate", 0.001)
    )
    model.compile(optimizer=optimizer, loss=loss)
    model.summary(print_fn=LOGGER.info)

    # prepare training
    if callbacks is None:
        callbacks = []
    if cfg.get("patience", None):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=cfg["patience"],
                monitor="val_loss",
                restore_best_weights=True,
            )
        )

    time_callback = TimeHistory()
    callbacks.append(time_callback)

    # see https://github.com/keras-team/keras/pull/16177
    if w_train_data is not None:
        w_train_data = pd.Series(w_train_data)

    LOGGER.info("Start training.")
    history = model.fit(
        x=datamodule.train.x,
        y=datamodule.train.y,
        # sample_weight=datamodule.train.w,
        epochs=cfg.get("epochs", 1),
        validation_data=(datamodule.val.x, datamodule.val.y),
        callbacks=callbacks,
        shuffle=cfg.get("shuffle", True),
        batch_size=cfg.get("batch_size", 1024),
        steps_per_epoch=cfg.get("steps_per_epoch", None),
        verbose=2,
    )
    LOGGER.info("Done! \U0001F40D")

    custom_objects = tf.keras.layers.serialize(model)
    if isinstance(loss_config, dict):
        loss_name = list(loss_config)[0]
    else:
        loss_name = loss_config
    custom_objects[loss_name] = loss
    history = history.history
    history["time"] = time_callback.times

    return model, custom_objects, datamodule.standardizer, history
