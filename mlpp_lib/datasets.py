from typing import Optional, Union

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
    *datasets: Union[xr.Dataset, None], event_dims: list
) -> list[Union[xr.DataArray, None]]:
    """Convert xarray datasets into tensors and filter out samples with missing values.
    If provided as an input, a None object is persisted in the output list."""

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


def split_dataset(
    dataset: xr.Dataset,
    splits: dict[str, dict],
    thinning: Optional[dict[str, int]] = None,
) -> dict[str, xr.Dataset]:
    """Split the dataset, optionally make the input dataset thinner."""
    if thinning:
        indexers = {dim: slice(None, None, step) for dim, step in thinning.items()}
    else:
        indexers = None
    return {
        key: dataset.sel(values).isel(indexers).load() for key, values in splits.items()
    }
