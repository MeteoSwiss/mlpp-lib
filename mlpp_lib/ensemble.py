from typing import Optional

import numpy as np
import xarray as xr  # type: ignore


def sortby(
    input_array: xr.DataArray,
    dim: str = "realization",
    loop_dim: Optional[str] = None,
    circular: bool = False,
    b: float = 0.0,
) -> xr.DataArray:
    """
    Sort by a dimension with the option of looping along a second dimension to
    reduce memory usage.

    Parameters
    ----------
    data_array: xr.DataArray
        Input data.
    dim: str
        Dimension along which to perform the sorting.
    loop_dim: str, optional
        If specified, sorting is done while looping along the given dimension
        to limit memory.
    circular: bool
        Sorting of circular data, i.e. angles between 0 and 360 degrees.
    b: float
        b parameter in the plotting positions formula. The default b=0 is equivalent to
        the Weibull method.

    Returns
    -------
    output_array: xr.DataArray
        Same as input_array, but sorted along dimension 'dim'.
    """
    if dim not in input_array.dims:
        raise ValueError(f"dimension {dim} does not exists")

    output_array = input_array.copy()

    if loop_dim is None:
        loop_dim = "dummy"

    dtype_input = output_array.dtype

    if circular:
        sin_sum = np.sin(output_array * np.pi / 180).sum(dim)
        cos_sum = np.cos(output_array * np.pi / 180).sum(dim)
        da_mean = np.arctan2(sin_sum, cos_sum) * 180 / np.pi + 360
        rank_origin = (da_mean + 180) % 360
        output_array = output_array.where(
            output_array >= rank_origin, output_array + 360
        )

    if loop_dim == "dummy":
        output_array = output_array.expand_dims("dummy")
    dai = []
    axis = list(output_array.dims).index(dim)
    for i in range(output_array[loop_dim].size):
        sub_data = output_array.isel({loop_dim: slice(i, i + 1)})
        random_noise = np.random.random(sub_data.shape) / 1e10
        ind = np.lexsort((random_noise, sub_data), axis=axis)
        dai.append(
            xr.apply_ufunc(
                np.take_along_axis,
                sub_data,
                kwargs={"indices": ind, "axis": axis},
            )
        )
    del random_noise
    output_array = xr.concat(dai, loop_dim, join="override")

    if loop_dim == "dummy":
        output_array = output_array.squeeze("dummy", drop=True)

    if circular:
        output_array = output_array % 360

    ens_size = output_array[dim].size
    ranks = np.arange(ens_size) + 1
    output_array[dim] = (ranks - b) / (ens_size + 1 - 2 * b)

    return output_array.rename({dim: "rank"}).astype(dtype_input)


def equidistant_resampling(
    input_dataset: xr.Dataset,
    ens_size: int,
    loop_dim: Optional[str] = None,
    b: float = 0.0,
    shuffle: bool = False,
) -> xr.Dataset:
    """Equidistant resampling of an ensemble to a lower ensemble size.

    Parameters
    ----------
    input_dataset: xr.Dataset
        Input dataset with dimension 'realization'.
    ens_size: int
        The target ensemble size, typically << than the original one.
    loop_dim: str, optional
        If specified, sorting is done while looping along the given dimension
        to limit memory.
    b: float, optional
        b parameter in the plotting positions formula. The default b=0 is equivalent to
        the Weibull method.
    shuffle: bool
        If False, it the output samples are sorted (default is False).

    Returns
    -------
    output_dataset: xr.Dataset
        Same as input_dataset, but with dimension 'realization' reduced to ens_size.
    """
    sel_ranks = np.arange(ens_size) + 1
    sel_ranks = (sel_ranks - b) / (ens_size + 1 - 2 * b)
    if shuffle:
        np.random.shuffle(sel_ranks)
    output_dataset = xr.Dataset()
    for name, data_array in input_dataset.data_vars.items():
        if "direction" in name:
            circular = True
        else:
            circular = False
        da_sorted = sortby(
            data_array,
            dim="realization",
            loop_dim=loop_dim,
            circular=circular,
            b=b,
        )
        output_dataset[name] = da_sorted.sel(rank=sel_ranks, method="nearest").rename(
            {"rank": "realization"}
        )

    return output_dataset


def compute_ecc(
    input_dataset: xr.Dataset,
    ecc_ranks: xr.DataArray,
    loop_dim: Optional[str] = None,
    b: float = 0.0,
) -> xr.Dataset:
    """Apply ECC

    Parameters
    ----------
    input_dataset: xr.Dataset
        Input dataset with dimension 'realization'.
    ecc_ranks: xr.DataArray
    loop_dim: str, optional
        If specified, sorting is done while looping along the given dimension
        to limit memory.
    b: float, optional
        b parameter in the plotting positions formula. The default b=0 is equivalent to
        the Weibull method.

    Returns
    -------
    output_dataset: xr.Dataset
    """
    align_dims = [dim for dim in input_dataset.dims.keys() if dim in ecc_ranks.dims]
    align_dims.remove("realization")
    ecc_ranks = ecc_ranks.sel({dim: input_dataset[dim] for dim in align_dims})

    # apply Weibull plotting positions to template ranks
    ens_size_template = ecc_ranks.realization.size
    ecc_ranks = (ecc_ranks + b) / (ens_size_template + 1 - 2 * b)

    # sort predictions
    output = xr.Dataset()
    for name, data_array in input_dataset.data_vars.items():
        if "direction" in name:
            circular = True
        else:
            circular = False
        da_sorted = sortby(
            data_array, dim="realization", loop_dim=loop_dim, circular=circular, b=b
        )
        # reorder predictions accordingly
        output[name] = da_sorted.sel(rank=ecc_ranks, method="nearest").drop("rank")

    return output
