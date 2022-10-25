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


def get_tensor_dataset(*datasets: xr.Dataset, event_dims: list) -> list[xr.DataArray]:
    """Convert xarray datasets into tensors and filter out samples with missing values"""

    tensors = []
    mask = True
    for i, dataset in enumerate(datasets):
        tensors.append(dataset_to_tensor(dataset, event_dims))
        mask_dataset_dims = [dim for dim in tensors[i].dims if dim != "sample"]
        mask *= np.isfinite(tensors[i]).all(dim=mask_dataset_dims)

    for j, tensor in enumerate(tensors):
        tensors[j] = tensor[mask]

    return tensors
