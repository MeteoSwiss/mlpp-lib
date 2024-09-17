import logging
from pprint import pformat
from typing import Optional

import tensorflow as tf

from mlpp_lib.callbacks import TimeHistory, EnsembleMetrics
from mlpp_lib.datasets import DataLoader, DataModule

from mlpp_lib.utils import (
    get_callback,
    get_loss,
    get_metric,
    get_model,
    get_optimizer,
    process_out_bias_init,
)


LOGGER = logging.getLogger(__name__)


def get_log_params(param_run: dict) -> dict:
    """Extract a selection of parameters for pretty logging"""
    log_params = {}
    # log_params["features_names"] = param_run["features"]
    # Note to future self: the list of features can easily exceed the maximum length for
    # a logged parameter, so we excluded it. Instead, this is logged as an artifact
    # together with the input run parameters.
    log_params["targets_names"] = param_run["targets"]
    log_params["sample_weights_names"] = param_run.get("sample_weights")
    log_params["event_dims"] = param_run["batching"]["event_dims"]
    log_params["thinning"] = param_run.get("thinning")
    log_params["model_name"] = list(param_run["model"])[0]
    log_params.update(param_run["model"][log_params["model_name"]])
    log_params["loss"] = param_run["loss"]
    return log_params


def get_lr(optimizer: tf.keras.optimizers.Optimizer) -> float:
    """Get the learning rate of the optimizer"""
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


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
    event_dims = list(set(datamodule.x.dims) - set(datamodule.batch_dims))
    out_bias_init = process_out_bias_init(
        datamodule.train.x, cfg.get("out_bias_init", "zeros"), event_dims
    )
    model_config[list(model_config)[0]].update({"out_bias_init": out_bias_init})
    input_shape = datamodule.train.x.shape[1:]
    output_shape = datamodule.train.y.shape[1:]
    model = get_model(input_shape, output_shape, model_config)
    loss = get_loss(loss_config)
    metrics = [get_metric(metric) for metric in cfg.get("metrics", [])]
    optimizer = get_optimizer(cfg.get("optimizer", "Adam"))
    metrics.append(get_lr(optimizer))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(print_fn=LOGGER.info)

    # callbacks
    if callbacks is None:
        callbacks = []

    for callback in cfg.get("callbacks", []):
        callback_instance = get_callback(callback)

        if isinstance(callback_instance, EnsembleMetrics):
            callback_instance.add_validation_data((datamodule.val.x, datamodule.val.y))

        callbacks.append(callback_instance)

    time_callback = TimeHistory()
    callbacks.append(time_callback)

    batch_size = cfg.get("batch_size", 1024)
    if datamodule.group_samples is not None:
        block_size = list(datamodule.group_samples.values())[0]
    else:
        block_size = 1
    # modify batch_size so that it becomes a multiple of block_size
    batch_size = (batch_size // block_size) * block_size

    train_dataloader = DataLoader(
        datamodule.train,
        batch_size=batch_size,
        shuffle=cfg.get("shuffle", True),
        block_size=block_size,
    )
    val_dataloader = DataLoader(
        datamodule.val,
        batch_size=batch_size,
        shuffle=False,
        block_size=block_size,
    )

    LOGGER.info("Start training.")
    LOGGER.debug(f"Length train data: {len(train_dataloader)}")
    LOGGER.debug(f"Length val data: {len(val_dataloader)}")
    res = model.fit(
        train_dataloader,
        epochs=cfg.get("epochs", 1),
        validation_data=val_dataloader,
        callbacks=callbacks,
        steps_per_epoch=cfg.get("steps_per_epoch", None),
        verbose=2,
    )
    LOGGER.info("Done! \U0001F40D")

    # we don't need to export loss and metric functions for deployments
    model.compile(optimizer=optimizer, loss=None, metrics=None)

    custom_objects = tf.keras.layers.serialize(model)

    history = res.history
    # for some reasons, 'lr' is provided as float32
    # and needs to be casted in order to be serialized
    for k in history:
        history[k] = list(map(float, history[k]))

    return model, custom_objects, datamodule.normalizer, history
