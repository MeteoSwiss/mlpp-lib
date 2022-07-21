import logging
from typing import Any, Callable, Union, Optional

import numpy as np
import xarray as xr
import tensorflow as tf

from mlpp_lib import losses, models


LOGGER = logging.getLogger(__name__)


def get_model(
    input_shape: tuple[int],
    output_shape: tuple[int],
    model_config: dict[str, Any],
) -> tf.keras.Model:
    """Get the keras model."""

    model_name = list(model_config.keys())[0]
    model_options = model_config[model_name]

    LOGGER.info(f"Using model: {model_name}")
    LOGGER.info(f"Input shape: {input_shape}, output shape: {output_shape}")
    LOGGER.debug(model_options)

    model = getattr(models, model_name)(input_shape, output_shape[-1], **model_options)

    return model


def get_loss(loss: Union[str, dict]) -> Callable:
    """Get the loss function, either keras built-in or mlpp custom."""

    if isinstance(loss, dict):
        loss_name = list(loss.keys())[0]
        loss_options = loss[loss_name]
    else:
        loss_name = loss

    if hasattr(losses, loss_name):
        LOGGER.info(f"Using custom-defined mlpp metric: {loss_name}")
        loss_obj = getattr(losses, loss_name)
        loss = loss_obj(**loss_options) if isinstance(loss_obj, type) else loss_obj
    elif hasattr(tf.keras.losses, loss_name):
        LOGGER.info(f"Using keras built-in metric: {loss_name}")
        loss_obj = getattr(tf.keras.losses, loss_name)
        loss = loss_obj(**loss_options) if isinstance(loss_obj, type) else loss_obj
    else:
        raise KeyError(f"The provided metric {loss} is not available.")

    return loss


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
