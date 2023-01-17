from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Optional, Sequence, Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Self

from .model_selection import DataSplitter


class DataModule:
    """A class to encapsulate everything involved in mlpp data processing.

    1. Load `xarray` objects from `.zarr` archives
    2. Filter, split, standardize.
    3. Load into mlpp `Dataset`
    4. Reshape/mask as needed.
    5. Serve the `Dataset` or wrap it inside a `DataLoader`

    Parameters
    ----------
    data_dir: string
        Path to the directory containing the raw data.
    features: list of strings
        Names of the predictors.
    targets: list of strings
        Names of the targets.
    batch_dims: list of strings
        Names of the dimensions that will be stacked into batches.
    splitter: `DataSplitter`
        The object that handles the data partitioning.

    """

    def __init__(
        self,
        data_dir: str,
        features: Sequence[Hashable],
        targets: Sequence[Hashable],
        batch_dims: Sequence[Hashable],
        splitter: DataSplitter,
        filter: Optional[DataFilter] = None,
        standardizer: Optional[Standardizer] = None,
        device: str = "/GPU:0",
    ):

        self.data_dir = data_dir
        self.features = features
        self.targets = targets
        self.splitter = splitter
        self.standardizer = standardizer
        self.batch_dims = batch_dims
        self.device = device

    def setup(self, stage=None):

        # load and preproc
        self.load_raw()
        self.select_splits(stage=stage)
        # self.apply_filter()
        self.standardize(stage=stage)

        # as datasets
        if stage == "fit" or stage is None:
            self.train = Dataset.from_xarray_datasets(*self.train).stack(
                self.batch_dims
            )
            self.val = Dataset.from_xarray_datasets(*self.val).stack(self.batch_dims)
        elif stage == "test" or stage is None:
            self.test = Dataset.from_xarray_datasets(*self.test).stack(self.batch_dims)

    def load_raw(self):
        self.x = (
            xr.open_zarr(self.data_dir + "features.zarr")[self.features]
            .reset_coords(drop=True)
            .astype(np.float32)
        )
        self.y = (
            xr.open_zarr(self.data_dir + "targets.zarr")[self.targets]
            .reset_coords(drop=True)
            .astype(np.float32)
        )

    def select_splits(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = self.splitter.get_partition(self.x, self.y, partition="train")
            self.val = self.splitter.get_partition(self.x, self.y, partition="val")
        elif stage == "test" or stage is None:
            self.test = self.splitter.get_partition(self.x, self.y, partition="test")

    def standardize(self, stage=None):

        if self.standardizer is None:
            self.standardizer = Standardizer()
            self.standardizer.fit(self.train[0])

        if stage == "fit" or stage is None:
            self.train = (
                tuple(self.standardizer.transform(self.train[0])) + self.train[1:]
            )
            self.val = tuple(self.standardizer.transform(self.val[0])) + self.val[1:]
        elif stage == "test" or stage is None:
            self.test = tuple(self.standardizer.transform(self.test[0])) + self.test[1:]

    def train_dataloader(self, batch_size):
        return DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=True,
            device=self.device,
        )

    def val_dataloader(self, batch_size):
        return DataLoader(
            self.val,
            batch_size=batch_size,
            shuffle=False,
            device=self.device,
        )

    def test_dataloader(self, batch_size):
        return DataLoader(
            self.test,
            batch_size=batch_size,
            shuffle=False,
            device=self.device,
        )


class Dataset:
    """A helper class for handling mlpp data

    Facilitates transforming xarray-based mlpp data into stacked
    n-d tensors and vice-versa.

    Parameters
    ----------
    x: array-like
        The input data.
    y: array-like
        The target data.
    dims: list or tuple
        Names of the dimensions. Last dimension is always `v`.
    coords: mapping of strings to array-like
        The mlpp coordinates for this dataset `(forecast_refererence_time, t, station)`
    features: list of strings
        Names of the input predictors.
    targets:
        Names of the target predictands.
    """

    mask: Optional[np.ndarray | da.Array]
    batch_dims: Sequence[Hashable]

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dims: Sequence[Hashable],
        coords: dict[str, Sequence],
        features: Optional[Sequence[Hashable]] = None,
        targets: Optional[Sequence[Hashable]] = None,
    ):

        self.x = x
        self.y = y
        self.dims = dims
        self.coords = coords
        self.features = features
        self.targets = targets
        self._is_stacked = True if "s" in dims else False

    @classmethod
    def from_xarray_datasets(cls: Self, x: xr.Dataset, y: xr.Dataset) -> Self:
        """
        Create a mlpp `Dataset` from `xr.Datasets`.

        Parameters
        ----------
        x: xr.Dataset
            The dataset containing the input features.
        y: xr.Dataset
            The dataset containing the input targets.
        """
        x = x.compute()
        y = y.compute()
        dims, coords = cls._check_coords(x, y)
        dims.append("v")
        features = list(x.data_vars)
        targets = list(y.data_vars)
        x = x.to_array("v").transpose(..., "v").values
        y = y.to_array("v").transpose(..., "v").values
        return cls(x, y, dims, coords, features, targets)

    def stack(self, batch_dims: Optional[Sequence[Hashable]] = None) -> Self:
        """Stack batch dimensions along the first axis"""
        x, y = self._as_variables()
        dims = ["s"] + [dim for dim in x.dims if dim not in batch_dims]
        x = x.stack(s=batch_dims).transpose("s", ...).values.copy(order="C")
        y = y.stack(s=batch_dims).transpose("s", ...).values.copy(order="C")
        ds = Dataset(x, y, dims, self.coords, self.features, self.targets)
        ds.batch_dims = batch_dims
        ds._is_stacked = True
        del x, y
        return ds

    def unstack(self) -> Self:
        """Unstack batch dimensions"""
        x, y = self._as_variables()
        x = x.unstack(s={dim: len(coord) for dim, coord in self.coords.items()})
        y = y.unstack(s={dim: len(coord) for dim, coord in self.coords.items()})
        ds = Dataset(
            x.values, y.values, x.dims, self.coords, self.features, self.targets
        )
        ds.batch_dims = self.batch_dims
        ds._is_stacked = False
        return ds

    @staticmethod
    def _check_coords(x: xr.Dataset | xr.DataArray, y: xr.Dataset | xr.DataArray):

        if x.dims != y.dims:
            raise ValueError(
                "x and y do not have the same dimensions! "
                f"x has dimensions {x.dims} and y has {y.dims}"
            )
        else:
            dims = list(x.dims)

        coords = {}
        for dim in dims:
            if dim == "v":
                continue
            elif any(x[dim].values != y[dim].values):
                raise ValueError(
                    "x and y have different coordinatres on the " f"{dim} dimension!"
                )
            else:
                coords[dim] = x[dim].values
        return dims, coords

    def _as_dataarrays(self) -> tuple[xr.DataArray, ...]:
        features = {} if self.features is None else {"v": self.features}
        targets = {} if self.targets is None else {"v": self.targets}
        xcoords = None if self.coords is None else self.coords | features
        ycoords = None if self.coords is None else self.coords | targets
        x = xr.DataArray(self.x, coords=xcoords, dims=list(xcoords.keys()))
        y = xr.DataArray(self.y, coords=ycoords, dims=list(ycoords.keys()))
        return x, y

    def _as_variables(self) -> tuple[xr.Variable, ...]:
        x = xr.Variable(self.dims, self.x)
        y = xr.Variable(self.dims, self.y)
        return x, y

    def drop_nans(self):

        x = da.from_array(self.x, name="x")
        y = da.from_array(self.y, name="x")
        self.mask = ~(
            da.any(da.isnan(x), axis=-1) | da.any(da.isnan(y), axis=-1)
        ).compute()

        self.x = x[self.mask].compute()
        self.y = y[self.mask].compute()

    def get_multiindex(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            list(self.coords.values), names=list(self.coords.keys())
        )

    def dataset_from_predictions(self, preds: np.ndarray):
        out = np.full_like(self.y, fill_value=np.nan)
        out[self.mask] = preds
        out = xr.Variable(self.dims, out)
        out = out.unstack(s={dim: len(coord) for dim, coord in self.coords.items()})
        out = xr.DataArray(out, coords=self.coords | {"v": self.targets})
        return out.to_dataset("v")


class DataLoader:
    """A dataloader for mlpp

    Parameters
    ----------
    dataset: `Dataset`
        The `Dataset` object.
    batch_size: integer
        Size of each batch.
    shuffle: bool
        Enable or disable shuffling before each iteration.
    nan_handling: string
        How to handle missing values. Current options are "drop"


    Example:
    >>> train_dataloader = DataLoader(
    ...    train_dataset,
    ...    batch_size = 2056,
    ...    shuffle = True,
    ...    device = "/GPU:0",
    ... )
    ... for x, y in train_dataloader:
    ...     pred = model(x)
    ...     err = pred - y
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
    ):

        self.dataset = dataset

        self._drop_nans()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.dataset.x)
        self.num_batches = self.num_samples // batch_size
        # self._indices = tf.range(self.num_samples)
        self._seed = 0

        # if device:
        #     self._to_device(device)

        self._reset()

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, index):  # -> tuple[tf.Tensor, ...]:
        if index >= self.num_batches:
            # self._reset()
            raise IndexError
        start = index * self.batch_size
        end = index * self.batch_size + self.batch_size
        return self.dataset.x[start:end], self.dataset.y[start:end]

    # def _reset(self):
    #     """Reset iterator and shuffles data if needed"""
    #     self.index = 0
    #     if self.shuffle:
    #         self._indices = tf.random.shuffle(self._indices, seed=self._seed)
    #         self.dataset.x = tf.gather(self.dataset.x, self._indices)
    #         self.dataset.y = tf.gather(self.dataset.y, self._indices)
    #         self._seed += 1

    # def _to_device(self, device):
    #     """Transfer data to a device"""
    #     with tf.device(device):
    #         self.dataset.x = tf.constant(self.dataset.x)
    #         self.dataset.y = tf.constant(self.dataset.y)

    def _drop_nans(self):
        self.mask = ~(
            da.any(da.isnan(self.dataset.x), axis=-1)
            | da.any(da.isnan(self.dataset.y), axis=-1)
        )

        self.dataset.x = self.dataset.x[self.mask].compute()
        self.dataset.y = self.dataset.y[self.mask].compute()


class DataFilter:
    """
    A class to handle data filtering in mlpp.

    Parameters
    ----------
    qa_filter: string-like, optional
        Path to the `filtered_measurements.nc` file. Bad measurements are set to NaNs.
    idx_keep: dict
        Mapping of xarray's indexers with labels that will be selected.
    idx_drop: dict
        Mapping of xarray's indexers with labels that will be dropped.

    """

    def __init__(
        self,
        qa_filter: Optional[str] = None,
        idx_keep: Optional[dict[str, Any]] = None,
        idx_drop: Optional[dict[str, Any]] = None,
    ):

        self.qa_mask = xr.load_dataarray(qa_filter) if qa_filter is not None else None

    def apply_filters(self, x: xr.Dataset, y: xr.Dataset):
        y = y.where(~self.qa_mask)


@dataclass
class Standardizer:
    """
    Standardizes data in a xarray.Dataset object.
    """

    mean: xr.Dataset = field(init=False)
    std: xr.Dataset = field(init=False)
    fillvalue: dict[str, float] = field(init=False)

    def fit(self, dataset: xr.Dataset, dims: Optional[list] = None):
        self.mean = dataset.mean(dims).copy(deep=True).compute()
        self.std = dataset.std(dims).copy(deep=True).compute()
        self.fillvalue = -5
        # Check for near-zero standard deviations and set them equal to one
        self.std = xr.where(self.std < 1e-6, 1, self.std)

    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")

        def f(ds: xr.Dataset):
            for var in ds.data_vars:
                assert var in self.mean.data_vars, f"{var} not in Standardizer"
            return ((ds - self.mean) / self.std).astype("float32")

        return tuple([f(ds) for ds in datasets])

    def inverse_transform(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")
        for var in dataset.data_vars:
            assert var in self.mean.data_vars, f"{var} not in Standardizer"
        dataset = xr.where(dataset > self.fillvalue, dataset, np.nan)
        return (dataset * self.std + self.mean).astype("float32")

    def save_json(self, out_fn: str) -> None:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")

        out_dict = {
            "mean": self.mean.to_dict(),
            "std": self.std.to_dict(),
            "fillvalue": self.fillvalue,
        }
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)

    def load_json(self, in_fn: str) -> None:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)

        self.mean = xr.Dataset.from_dict(in_dict["mean"])
        self.std = xr.Dataset.from_dict(in_dict["std"])
        self.fillvalue = in_dict["fillvalue"]
