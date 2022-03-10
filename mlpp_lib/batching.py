import numpy as np
import xarray as xr
import tensorflow as tf


def get_tensor_dataset(
    features: xr.Dataset, targets: xr.Dataset, event_dims: list, batch_size: int
) -> tf.data.Dataset:

    stacked_dims = list(set(features.dims) - set(event_dims))

    x = (
        features.to_array("variable")
        .stack(sample=stacked_dims)
        .transpose("sample", ..., "variable")
    )
    y = (
        targets.to_array("target")
        .stack(sample=stacked_dims)
        .transpose("sample", ..., "target")
    )

    mask_x_dims = [dim for dim in x.dims if dim != "sample"]
    mask_y_dims = [dim for dim in y.dims if dim != "sample"]

    x = x[np.isfinite(y).all(dim=mask_y_dims)]
    y = y[np.isfinite(y).all(dim=mask_y_dims)]
    y = y[np.isfinite(x).all(dim=mask_x_dims)]
    x = x[np.isfinite(x).all(dim=mask_x_dims)]

    data = tf.data.Dataset.from_tensor_slices((x.values, y.values))
    data = data.shuffle(int(x.shape[0] / 2)).batch(batch_size)

    data.features = features
    data.targets = targets
    data.batch_size = batch_size
    data.event_dims = event_dims
    data.stacked_dims = stacked_dims

    return data
