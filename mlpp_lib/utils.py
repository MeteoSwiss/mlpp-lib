import logging
from typing import Any, Callable, Union, Optional

import numpy as np
import xarray as xr
import tensorflow as tf

from mlpp_lib import callbacks, losses, metrics, models


LOGGER = logging.getLogger(__name__)


def get_callback(callback: Union[str, dict]) -> Callable:
    """Get the callback, either keras built-in or mlpp custom."""

    if isinstance(callback, dict):
        callback_name = list(callback.keys())[0]
        callback_options = callback[callback_name]
    else:
        callback_name = callback
        callback_options = {}

    if hasattr(callbacks, callback_name):
        LOGGER.info(f"Using custom mlpp callback: {callback_name}")
        callback_obj = getattr(callbacks, callback_name)
        callback = (
            callback_obj(**callback_options)
            if isinstance(callback_obj, type)
            else callback_obj
        )
    elif hasattr(tf.keras.callbacks, callback_name):
        LOGGER.info(f"Using keras built-in callback: {callback_name}")
        callback_obj = getattr(tf.keras.callbacks, callback_name)
        callback = (
            callback_obj(**callback_options)
            if isinstance(callback_obj, type)
            else callback_obj
        )
    else:
        raise KeyError(f"The callback {callback_name} is not available.")

    return callback


def get_model(
    input_shape: tuple[int],
    output_shape: Union[int, tuple[int]],
    model_config: dict[str, Any],
) -> tf.keras.Model:
    """Get the keras model."""

    model_name = list(model_config.keys())[0]
    model_options = model_config[model_name]

    LOGGER.info(f"Using model: {model_name}")
    LOGGER.info(f"Input shape: {input_shape}, output shape: {output_shape}")
    LOGGER.debug(model_options)
    if isinstance(output_shape, int):
        output_shape = (output_shape,)
    model = getattr(models, model_name)(input_shape, output_shape[-1], **model_options)

    return model


def get_loss(loss: Union[str, dict]) -> Callable:
    """Get the loss function, either keras built-in or mlpp custom."""

    if isinstance(loss, dict):
        loss_name = list(loss.keys())[0]
        loss_options = loss[loss_name]
    else:
        loss_name = loss
        loss_options = {}

    if hasattr(losses, loss_name):
        LOGGER.info(f"Using custom mlpp loss: {loss_name}")
        loss_obj = getattr(losses, loss_name)
        loss = loss_obj(**loss_options) if isinstance(loss_obj, type) else loss_obj
    elif hasattr(tf.keras.losses, loss_name):
        LOGGER.info(f"Using keras built-in loss: {loss_name}")
        loss_obj = getattr(tf.keras.losses, loss_name)
        loss = loss_obj(**loss_options) if isinstance(loss_obj, type) else loss_obj
    else:
        raise KeyError(f"The loss {loss_name} is not available.")

    return loss


def get_metric(metric: Union[str, dict]) -> Callable:
    """Get the metric function, either keras built-in or mlpp custom."""

    if isinstance(metric, dict):
        metric_name = list(metric.keys())[0]
        metric_options = metric[metric_name]
    else:
        metric_name = metric
        metric_options = {}

    if hasattr(metrics, metric_name):
        LOGGER.info(f"Using custom mlpp metric: {metric_name}")
        metric_obj = getattr(metrics, metric_name)
        metric = (
            metric_obj(**metric_options) if isinstance(metric_obj, type) else metric_obj
        )
    elif hasattr(tf.keras.metrics, metric_name):
        LOGGER.info(f"Using keras built-in metric: {metric_name}")
        metric_obj = getattr(tf.keras.metrics, metric_name)
        metric = (
            metric_obj(**metric_options) if isinstance(metric_obj, type) else metric_obj
        )
    else:
        raise KeyError(f"The metric {metric_name} is not available.")

    return metric


def get_scheduler(
    scheduler_config: Union[dict, None]
) -> Optional[tf.keras.optimizers.schedules.LearningRateSchedule]:
    """Create a learning rate scheduler from a config dictionary."""

    if not isinstance(scheduler_config, dict):
        LOGGER.info("Not using a scheduler.")
        return None

    if len(scheduler_config) != 1:
        raise ValueError(
            "Scheduler configuration should contain exactly one scheduler name with its options."
        )

    scheduler_name = next(
        iter(scheduler_config)
    )  # first key is the name of the scheduler
    scheduler_options = scheduler_config[scheduler_name]

    if not isinstance(scheduler_options, dict):
        raise ValueError(
            f"Scheduler options for '{scheduler_name}' should be a dictionary."
        )

    if hasattr(tf.keras.optimizers.schedules, scheduler_name):
        LOGGER.info(f"Using keras built-in learning rate scheduler: {scheduler_name}")
        scheduler_cls = getattr(tf.keras.optimizers.schedules, scheduler_name)
        scheduler = scheduler_cls(**scheduler_options)
    else:
        raise KeyError(
            f"The scheduler '{scheduler_name}' is not available in tf.keras.optimizers.schedules."
        )

    return scheduler


def get_optimizer(optimizer: Union[str, dict]) -> Callable:
    """Get the optimizer, keras built-in only."""

    if isinstance(optimizer, dict):
        optimizer_name = list(optimizer.keys())[0]
        optimizer_options = optimizer[optimizer_name]
        if scheduler := get_scheduler(optimizer_options.pop("learning_rate", None)):
            optimizer_options["learning_rate"] = scheduler
    else:
        optimizer_name = optimizer
        optimizer_options = {}

    if hasattr(tf.keras.optimizers, optimizer_name):
        LOGGER.info(f"Using keras built-in optimizer: {optimizer_name}")
        optimizer_obj = getattr(tf.keras.optimizers, optimizer_name)
        optimizer = (
            optimizer_obj(**optimizer_options)
            if isinstance(optimizer_obj, type)
            else optimizer_obj
        )
    else:
        raise KeyError(f"The optimizer {optimizer_name} is not available.")

    return optimizer


def process_out_bias_init(
    data: xr.DataArray,
    out_bias_init: Optional[Union[str, np.ndarray[Any, float]]],
    event_dims: list,
) -> Union[str, np.ndarray[Any, float]]:
    """If needed, pre-compute the initial bias for the output layer."""
    out_bias_init = out_bias_init if out_bias_init else "zeros"
    if out_bias_init == "mean":
        out_bias_init = data.mean(dim=["sample", *event_dims]).values
    return out_bias_init


def as_weather(dataset: xr.Dataset) -> xr.Dataset:
    """Adjust/convert/process weather data to physical values. It removes nonphysical
    values (e.g. negative wind speeds) and converts sine/cosine values to wind directions,
    as well as northward/eastward components to speed and direction.

    It deals with both <var_name> and <source:var_name> notations.
    """
    new_set = xr.Dataset()
    for source_var in dataset.data_vars:

        try:
            source, var = source_var.split(":")
        except ValueError:
            source = ""
            var = source_var

        if var in ["wind_speed_of_gust", "wind_speed"]:
            new_set[source_var] = xr.where(
                dataset[source_var] < 0, 0, dataset[source_var]
            )

        elif var == "sin_wind_from_direction":
            new_set[f"{source}:wind_from_direction".strip(":")] = (
                np.arctan2(
                    dataset[f"{source}:sin_wind_from_direction".strip(":")],
                    dataset[f"{source}:cos_wind_from_direction".strip(":")],
                )
                * 180
                / np.pi
                + 2 * 360
            ) % 360

        elif var == "cos_wind_from_direction":
            continue

        elif var == "northward_wind":
            # Transform wind components to scalars
            new_set[f"{source}:wind_speed".strip(":")] = np.sqrt(
                np.square(dataset[f"{source}:northward_wind".strip(":")])
                + np.square(dataset[f"{source}:eastward_wind".strip(":")])
            )
            new_set[f"{source}:wind_from_direction".strip(":")] = (
                270
                - 180
                / np.pi
                * np.arctan2(
                    dataset[f"{source}:northward_wind".strip(":")],
                    dataset[f"{source}:eastward_wind".strip(":")],
                )
                + 2 * 360
            ) % 360

        elif var == "eastward_wind":
            continue

        elif var == "cloud_area_fraction":
            new_set[source_var] = np.clip(dataset[source_var], 0, 1)

        else:
            new_set[source_var] = dataset[source_var]

    return new_set
