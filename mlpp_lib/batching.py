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
    *datasets: xr.Dataset or None, event_dims: list
) -> list[xr.DataArray or None]:
    """Convert xarray datasets into tensors and filter out samples with missing values.
    If provided as inputs, None as persisted in the output list."""

    tensors = []
    mask = True
    for i, dataset in enumerate(datasets):
        if dataset is None:
            tensors.append(None)
            continue
        tensors.append(dataset_to_tensor(dataset, event_dims))
        mask_dataset_dims = [dim for dim in tensors[i].dims if dim != "sample"]
        mask *= np.isfinite(tensors[i]).all(dim=mask_dataset_dims)

    for j, tensor in enumerate(tensors):
        if tensor is None:
            continue
        tensors[j] = tensor[mask]

    return tensors
