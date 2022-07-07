import numpy as np
import xarray as xr


def dataset_to_tensor(dataset: xr.Dataset, event_dims: list = []) -> xr.DataArray:
    """Convert a feature dataset into a N-D tensor"""
    stacked_dims = list(set(dataset.dims) - set(event_dims))
    tensor = (
        dataset.to_array("variable")
        .stack(sample=stacked_dims)
        .transpose("sample", ..., "variable")
    )
    return tensor


def get_tensor_dataset(
    features: xr.Dataset, targets: xr.Dataset, event_dims: list
) -> tuple[xr.DataArray]:

    features_tensor = dataset_to_tensor(features, event_dims)
    targets_tensor = dataset_to_tensor(targets, event_dims)

    mask_features_dims = [dim for dim in features_tensor.dims if dim != "sample"]
    mask_targets_dims = [dim for dim in targets_tensor.dims if dim != "sample"]

    features_tensor = features_tensor[
        np.isfinite(targets_tensor).all(dim=mask_targets_dims)
    ]
    targets_tensor = targets_tensor[
        np.isfinite(targets_tensor).all(dim=mask_targets_dims)
    ]
    targets_tensor = targets_tensor[
        np.isfinite(features_tensor).all(dim=mask_features_dims)
    ]
    features_tensor = features_tensor[
        np.isfinite(features_tensor).all(dim=mask_features_dims)
    ]

    return features_tensor, targets_tensor
