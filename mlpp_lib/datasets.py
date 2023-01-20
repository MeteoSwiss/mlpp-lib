from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Optional, Sequence, Union, Callable

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Self

from .model_selection import DataSplitter
from .standardizers import Standardizer


class DataModule:
    """A class to encapsulate everything involved in mlpp data processing.

    1. Load `xarray` objects from `.zarr` archives and select variables.
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
        sample_weighting: Optional[Sequence[Hashable] | Hashable] = None,
        thinning: Optional[Mapping[str, Sequence]] = None,
        device: Optional[str] = None,
    ):

        self.data_dir = data_dir
        self.features = features
        self.targets = targets
        self.batch_dims = batch_dims
        self.splitter = splitter
        self.filter = filter
        self.standardizer = standardizer
        self.sample_weighting = (
            list(sample_weighting) if sample_weighting is not None else None
        )
        self.thinning = thinning
        self.device = device

    def setup(self, stage=None):

        # load and preproc
        self.load_raw()
        self.select_splits(stage=stage)
        if self.filter is not None:
            self.apply_filter(stage=stage)
        self.standardize(stage=stage)

        if stage == "fit" or stage is None:
            self.train = (
                Dataset.from_xarray_datasets(*self.train)
                .stack(self.batch_dims)
                .drop_nans()
            )
            self.val = (
                Dataset.from_xarray_datasets(*self.val)
                .stack(self.batch_dims)
                .drop_nans()
            )
        elif stage == "test" or stage is None:
            self.test = (
                Dataset.from_xarray_datasets(*self.test)
                .stack(self.batch_dims)
                .drop_nans()
            )

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

        if w := self.sample_weighting:
            try:
                self.w = (
                    xr.open_zarr(self.data_dir + "features.zarr")[w]
                    .reset_coords(drop=True)
                    .astype(np.float32)
                )
            except KeyError:
                self.w = (
                    xr.open_zarr(self.data_dir + "targets.zarr")[w]
                    .reset_coords(drop=True)
                    .astype(np.float32)
                )
        else:
            self.w = None

    def select_splits(self, stage=None):
        args = (self.x, self.y, self.w) if self.w is not None else (self.x, self.y)
        if stage == "fit" or stage is None:
            self.train = self.splitter.get_partition(
                *args, partition="train", thinning=self.thinning
            )
            self.val = self.splitter.get_partition(
                *args, partition="val", thinning=self.thinning
            )
        elif stage == "test" or stage is None:
            self.test = self.splitter.get_partition(
                *args, partition="test", thinning=self.thinning
            )

    def apply_filter(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = self.filter.apply(*self.train)
            self.val = self.filter.apply(*self.val)
        if stage == "test" or stage is None:
            self.test = self.filter.apply(*self.test)

    def standardize(self, stage=None):

        if self.standardizer is None:
            if stage == "test":
                raise ValueError("Must provide standardizer for `test` stage.")

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

    batch_dims: Sequence[Hashable]

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dims: Sequence[Hashable],
        coords: dict[str, Sequence],
        features: Optional[Sequence[Hashable]] = None,
        targets: Optional[Sequence[Hashable]] = None,
        w: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):

        self.x = x
        self.y = y
        self.w = w
        self.mask = mask
        self.dims = dims
        self.coords = coords
        self.features = features
        self.targets = targets
        self._is_stacked = True if "s" in dims else False

    @classmethod
    def from_xarray_datasets(
        cls: Self, x: xr.Dataset, y: xr.Dataset, w: Optional[xr.Dataset] = None
    ) -> Self:
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
        if "is_valid" in list(x._coord_names):
            x = x.where(x.coords["is_valid"])  # TODO: this can be optimized
        dims, coords = cls._check_coords(x, y)
        dims.append("v")
        features = list(x.data_vars)
        targets = list(y.data_vars)
        x = x.to_array("v").transpose(..., "v").values
        y = y.to_array("v").transpose(..., "v").values
        if w is not None:
            w = w.to_array("v").transpose(..., "v").values
            w = np.prod(w, axis=-1)
        return cls(x, y, dims, coords, features, targets, w=w)

    def stack(self, batch_dims: Optional[Sequence[Hashable]] = None) -> Self:
        """Stack batch dimensions along the first axis"""
        x, y, w = self._as_variables()
        dims = ["s"] + [dim for dim in x.dims if dim not in batch_dims]
        x = x.stack(s=batch_dims).transpose("s", ...).values.copy(order="C")
        y = y.stack(s=batch_dims).transpose("s", ...).values.copy(order="C")
        if w is not None:
            w = w.stack(s=batch_dims).transpose("s", ...).values.copy(order="C")
        ds = Dataset(x, y, dims, self.coords, self.features, self.targets, w=w)
        ds.batch_dims = batch_dims
        ds._is_stacked = True
        del x, y, w
        return ds

    def unstack(self) -> Self:
        """Unstack batch dimensions TODO: this is messy"""
        if not self._is_stacked:
            raise ValueError("Nothing to unstack!")

        if self.mask is not None:
            x, y, w = self._undrop_nans()
            dims = (*self.coords.keys(), "v")
            ds = Dataset(x, y, dims, self.coords, self.features, self.targets, w=w)
            del x, y, w 
            return ds 
        else:
            x, y, w = self._as_variables()
            unstack_info = {dim: len(coord) for dim, coord in self.coords.items()}
            x = x.unstack(s=unstack_info).transpose(..., "v").values
            y = y.unstack(s=unstack_info).transpose(..., "v").values
            dims = (*self.coords.keys(), "v")
            if w is not None:
                w = w.unstack(s=unstack_info).values
            ds = Dataset(x, y, dims, self.coords, self.features, self.targets, w=w)
            ds.batch_dims = self.batch_dims
            ds._is_stacked = False
            del x, y, w
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
        if self.w:
            w = xr.DataArray(self.w, coords=self.coords, dims=list(self.coords))
        else:
            w = None
        return x, y, w

    def _as_variables(self) -> tuple[xr.Variable, ...]:
        x = xr.Variable(self.dims, self.x)
        y = xr.Variable(self.dims, self.y)
        w = xr.Variable(self.dims[:-1], self.w) if self.w is not None else None
        return x, y, w

    def drop_nans(self):
        if not self._is_stacked:
            raise ValueError("Dataset shoud be stacked before dropping samples.")

        x, y, w = self._get_copies()

        mask = ~(
            da.any(da.isnan(da.from_array(x, name="x")), axis=-1) 
            | da.any(da.isnan(da.from_array(y, name="y")), axis=-1)
        ).compute()

        x = x[mask]
        y = y[mask]
        w = w[mask] if w is not None else None

        ds = Dataset(
            x, y, self.dims, self.coords, self.features, self.targets, w=w, mask=mask
        )
        ds._is_stacked = self._is_stacked
        ds.batch_dims = self.batch_dims
        del x, y, w, mask
        return ds

    def _undrop_nans(self):

        original_shape = tuple(len(c) for c in self.coords.values())
        x, y, w = self._get_copies()
        x_tmp = np.full((*original_shape, len(self.features)), fill_value=np.nan)
        y_tmp = np.full((*original_shape, len(self.targets)), fill_value=np.nan)
        mask = self.mask.reshape(*original_shape).copy()
        x_tmp[mask] = x
        y_tmp[mask] = y
        x = x_tmp
        y = y_tmp
        del x_tmp, y_tmp

        if w is not None:
            w_tmp = np.full(
                original_shape, fill_value=np.nan
            )  # https://github.com/dask/dask/issues/8460
            w_tmp[mask] = w
            w = w_tmp.copy()
            del w_tmp
        
        return x, y, w 

    def get_multiindex(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            list(self.coords.values), names=list(self.coords.keys())
        )

    def dataset_from_predictions(self, preds: np.ndarray) -> xr.Dataset:
        out = np.full_like(self.y, fill_value=np.nan)
        out[self.mask] = preds
        out = xr.Variable(self.dims, out)
        out = out.unstack(s={dim: len(coord) for dim, coord in self.coords.items()})
        out = xr.DataArray(out, coords=self.coords | {"v": self.targets})
        return out.to_dataset("v")

    def _get_copies(self):
        x = self.x.copy()
        y = self.y.copy()
        w = self.w.copy() if self.w is not None else None
        del self.x, self.y, self.w
        return x, y, w 


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
        Path to the `filtered_measurements.nc` file. Bad measurements (y) will be set to NaNs.
    x_filter: callable, optional
        Function that takes the input features `xr.Dataset` object as input and return a
        boolean mask in form of a `xr.DataArray`, which will be attached to the input dataset
        as a new coordinate called `is_valid`.
    """

    def __init__(
        self,
        qa_filter: Optional[str] = None,
        x_filter: Optional[Callable] = None,
    ):

        self.qa_mask = xr.load_dataarray(qa_filter) if qa_filter is not None else None
        self.x_filter = x_filter

    def apply(self, x: xr.Dataset, y: xr.Dataset, w: Optional[xr.Dataset] = None):

        if self.qa_mask is not None:
            y = y.where(~self.qa_mask)

        if self.x_filter is not None:
            x = x.assign_coords(is_valid=self.x_filter(x))

        out = (x, y) if w is None else (x, y, w)

        return out
