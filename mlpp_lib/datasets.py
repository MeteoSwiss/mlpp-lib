from __future__ import annotations

import logging
from typing import Hashable, Mapping, Optional, Sequence, Callable

import dask.array as da
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from typing_extensions import Self

from .model_selection import DataSplitter
from .normalizers import DataTransformer

LOGGER = logging.getLogger(__name__)


class DataModule:
    """A class to encapsulate everything involved in mlpp data processing.

    1. Take xarray objects or load and select variables from `.zarr` archives.
    2. Filter, split, normalize.
    3. Load into mlpp `Dataset`
    4. Reshape/mask as needed.
    5. Serve the `Dataset` or wrap it inside a `DataLoader`

    Parameters
    ----------
    features: list of strings or xr.Dataset
        Names of the predictors or an xr.Dataset containing the predictors.
    targets: list of strings
        Names of the targets or an xr.Dataset containing the targets.
    batch_dims: list of strings
        Names of the dimensions that will be stacked into batches.
    splitter: `DataSplitter`
        The object that handles the data partitioning.
    group_samples: dict, optional
        Mapping of the form `{dim_name: group_size}` to indicate that the original order
        of the samples should be conserved for the `dim_name` dimension when stacking.
        The dimension `dim_name` must be one of the batch dimensions in `batch_dims`.
        If `group_samples` is not None, NaNs are removed in blocks of size `group_size`.
    data_dir: string
        Path to the directory containing the raw data. Must be provided if features
        and targets are lists of names.
    filter: `DataFilter`, optional
        The object that handles the data filtering.
    normalizer: `DataTransformer`, optional
        The object to normalize data.
        Must be provided if `.setup("test")` is called.
    sample_weighting: list of str or str or xr.Dataset, optional
        Name(s) of the variable(s) used for weighting dataset samples or an xr.Dataset
        containing the sample weights.
    thinning: mapping, optional
        Thinning factor as integer for a dimension.
    """

    def __init__(
        self,
        features: Sequence[Hashable] or xr.Dataset,
        targets: Sequence[Hashable] or xr.Dataset,
        batch_dims: Sequence[Hashable],
        splitter: DataSplitter,
        group_samples: Optional[dict[str:int]] = None,
        data_dir: Optional[str] = None,
        filter: Optional[DataFilter] = None,
        normalizer: Optional[DataTransformer] = None,
        sample_weighting: Optional[Sequence[Hashable] or Hashable or xr.Dataset] = None,
        thinning: Optional[Mapping[str, int]] = None,
    ):

        self.data_dir = data_dir
        self.features = features
        self.targets = targets
        self.batch_dims = batch_dims
        self.splitter = splitter
        self.group_samples = group_samples
        self.filter = filter
        self.normalizer = normalizer
        self.sample_weighting = (
            list(sample_weighting)
            if isinstance(sample_weighting, str)
            else sample_weighting
        )
        self.thinning = thinning

    def setup(self, stage=None):
        """Prepare the datamodule for fitting, testing or both.

        Parameters
        ----------
        stage: str, optional
            If "fit", only training and validation data are processed.
            If "test", only testing data is processed.
            If None, all data is processed.
        """
        LOGGER.info(f"Beginning datamodule setup for stage='{stage}'")
        maybe_load = self._check_args()
        if maybe_load:
            self.load_raw()
        if self.filter is not None:
            self.apply_filter()
        self.select_splits(stage=stage)
        self.normalize(stage=stage)
        self.as_datasets(stage=stage)

    def load_raw(self):
        LOGGER.info(f"Loading data from: {self.data_dir}")

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
        LOGGER.info("Selecting splits.")
        args = (self.x, self.y, self.w) if self.w is not None else (self.x, self.y)
        if stage == "fit" or stage is None:
            self.train = self.splitter.get_partition(
                *args, partition="train", thinning=self.thinning
            )
            self.val = self.splitter.get_partition(
                *args, partition="val", thinning=self.thinning
            )
        if stage == "test" or stage is None:
            self.test = self.splitter.get_partition(
                *args, partition="test", thinning=self.thinning
            )

    def apply_filter(self):
        LOGGER.info("Applying filter to features and targets.")
        self.x, self.y = self.filter.apply(self.x, self.y)

    def normalize(self, stage=None):
        LOGGER.info("Normalizing data.")

        if self.normalizer is None:
            if stage == "test":
                raise ValueError("Must provide normalizer for `test` stage.")
            else:
                LOGGER.warning("No normalizer found, data are standardized by default.")
                self.normalizer = DataTransformer(
                    {"Standardizer": list(self.train[0].data_vars)}
                )

        if stage == "fit" or stage is None:
            self.normalizer.fit(self.train[0])
            self.train = (
                tuple(self.normalizer.transform(self.train[0])) + self.train[1:]
            )
            self.val = tuple(self.normalizer.transform(self.val[0])) + self.val[1:]
        if stage == "test" or stage is None:
            self.test = tuple(self.normalizer.transform(self.test[0])) + self.test[1:]

    def as_datasets(self, stage=None):
        batch_dims = self.batch_dims
        if self.group_samples:
            group_dim, group_size = list(self.group_samples.items())[0]
            if group_dim not in batch_dims:
                raise ValueError(
                    f"The dimension {group_dim} used for grouping samples "
                    f"must be one of the batch dimensions: {batch_dims}"
                )
            batch_dims.remove(group_dim)
            batch_dims.insert(0, group_dim)
        else:
            group_size = 1

        LOGGER.info("Dask is computing...")
        if stage == "fit" or stage is None:
            self.train = (
                Dataset.from_xarray_datasets(*self.train)
                .stack(batch_dims)
                .drop_nans(group_size)
            )

            self.val = (
                Dataset.from_xarray_datasets(*self.val)
                .stack(batch_dims)
                .drop_nans(group_size)
            )
            LOGGER.info(f"Training dataset: {self.train}")
            LOGGER.info(f"Validation dataset: {self.val}")
        if stage == "test" or stage is None:
            self.test = (
                Dataset.from_xarray_datasets(*self.test)
                .stack(batch_dims)
                .drop_nans(group_size)
            )
            LOGGER.info(f"Test dataset: {self.test}")

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

    def _check_args(self):
        if isinstance(self.features, xr.Dataset):
            assert isinstance(self.targets, xr.Dataset)
            if self.sample_weighting is not None:
                assert isinstance(self.sample_weighting, xr.Dataset)
                self.w = self.sample_weighting.copy()
                self.sample_weighting = list(self.w.data_vars)
            else:
                self.w = self.sample_weighting
            self.x = self.features.copy()
            self.y = self.targets.copy()
            self.features = list(self.x.data_vars)
            self.targets = list(self.y.data_vars)
            maybe_load = False
        else:
            assert self.data_dir is not None
            if "zarr" not in xr.backends.list_engines():
                raise ModuleNotFoundError(
                    "zarr must be installed to read data from disk!"
                )
            maybe_load = True
        return maybe_load


class Dataset:
    """A helper class for handling mlpp data

    Facilitates transforming xarray-based mlpp data into stacked
    n-d tensors and vice-versa.

    Parameters
    ----------
    x: array-like
        The input data.
    dims: list or tuple
        Names of the dimensions. Last dimension is always `v`.
    coords: mapping of strings to array-like
        The mlpp coordinates for this dataset `(forecast_refererence_time, t, station)`
    features: list of strings
        Names of the input predictors.
    targets:
        Names of the target predictands.
    y: array-like, optional
        The target data.
    w: array-like, optional
        The sample weights data.
    mask: boolean array, optional
        The samples mask.
    """

    batch_dims: Sequence[Hashable]

    def __init__(
        self,
        x: np.ndarray,
        dims: Sequence[Hashable],
        coords: dict[str, Sequence],
        features: Optional[Sequence[Hashable]] = None,
        targets: Optional[Sequence[Hashable]] = None,
        y: Optional[np.ndarray] = None,
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
        cls: Self,
        x: xr.Dataset,
        y: Optional[xr.Dataset] = None,
        w: Optional[xr.Dataset] = None,
    ) -> Self:
        """
        Create a mlpp `Dataset` from `xr.Datasets`.

        Parameters
        ----------
        x: xr.Dataset
            The dataset containing the input features.
        y: xr.Dataset, optional
            The dataset containing the input targets.
        w: xr.Dataset, optional
            The dataset containing the variables to compute samples weights.

        Returns
        -------
        ds: Dataset
            The dataset instance.
        """
        x = x.compute()
        if "is_valid" in list(x._coord_names):
            x = x.where(x.coords["is_valid"])  # TODO: this can be optimized
        dims, coords = cls._check_coords(x, y)
        dims.append("v")
        features = list(x.data_vars)
        x = x.to_array("v").transpose(..., "v").values
        if y is not None:
            y = y.compute()
            targets = list(y.data_vars)
            y = y.to_array("v").transpose(..., "v").values
        else:
            targets = None
        if w is not None:
            w = w.fillna(1)
            w = w.to_array("v").transpose(..., "v").values
            w = np.prod(w, axis=-1)
        return cls(x, dims, coords, features, targets, y=y, w=w)

    def stack(self, batch_dims: Sequence[Hashable]) -> Self:
        """Stack batch dimensions along the first axis"""
        x, y, w = self._as_variables()
        dims = ["s"] + [dim for dim in x.dims if dim not in batch_dims]
        x = x.stack(s=batch_dims).transpose("s", ...).values.copy(order="C")
        if y is not None:
            y = y.stack(s=batch_dims).transpose("s", ...).values.copy(order="C")
        if w is not None:
            w = w.stack(s=batch_dims).transpose("s", ...).values.copy(order="C")
        ds = Dataset(x, dims, self.coords, self.features, self.targets, y=y, w=w)
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
            ds = Dataset(x, dims, self.coords, self.features, self.targets, y=y, w=w)
            del x, y, w
            return ds
        else:
            x, y, w = self._as_variables()
            unstack_info = {dim: len(coord) for dim, coord in self.coords.items()}
            x = x.unstack(s=unstack_info).transpose(..., "v").values
            dims = (*self.coords.keys(), "v")
            if y is not None:
                y = y.unstack(s=unstack_info).transpose(..., "v").values
            if w is not None:
                w = w.unstack(s=unstack_info).values
            ds = Dataset(x, dims, self.coords, self.features, self.targets, y=y, w=w)
            ds.batch_dims = self.batch_dims
            ds._is_stacked = False
            del x, y, w
            return ds

    @staticmethod
    def _check_coords(
        x: xr.Dataset or xr.DataArray, y: Optional[xr.Dataset or xr.DataArray] = None
    ):
        if y is None:
            dims = list(x.dims)
            coords = {dim: x[dim].values for dim in dims if dim != "v"}
            return dims, coords

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
                    "x and y have different coordinates on the " f"{dim} dimension!"
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
        y = xr.Variable(self.dims, self.y) if self.y is not None else None
        w = xr.Variable(self.dims[:-1], self.w) if self.w is not None else None
        return x, y, w

    def drop_nans(self, group_size: int = 1):
        """Drop incomplete samples and return a new `Dataset` with a mask."""
        if not self._is_stacked:
            raise ValueError("Dataset should be stacked before dropping samples.")

        x, y, w = self._get_copies()

        event_axes = [self.dims.index(dim) for dim in self.dims if dim != "s"]
        mask = da.any(da.isnan(da.from_array(x, name="x")), axis=event_axes)
        if y is not None:
            mask = mask | da.any(da.isnan(da.from_array(y, name="y")), axis=event_axes)
        mask = (~mask).compute()

        # with grouped samples, nans have to be removed in blocks:
        # if one or more nans are found in a given block, the entire block is dropped
        if group_size > 1:
            mask_length = len(mask)
            remainder = mask_length % group_size
            pad_length = 0 if remainder == 0 else group_size - remainder
            padded_mask = np.pad(
                mask, (0, pad_length), mode="constant", constant_values=True
            )
            grouped_mask = np.amin(padded_mask.reshape(-1, group_size), axis=1)
            mask = grouped_mask.repeat(group_size)[:mask_length]

        x = x[mask]
        y = y[mask] if y is not None else None
        w = w[mask] if w is not None else None

        ds = Dataset(
            x, self.dims, self.coords, self.features, self.targets, y=y, w=w, mask=mask
        )
        ds._is_stacked = self._is_stacked
        ds.batch_dims = self.batch_dims
        del x, y, w, mask
        return ds

    def _undrop_nans(self):

        original_shape = tuple(len(c) for c in self.coords.values())
        mask = self.mask.reshape(*original_shape).copy()
        x, y, w = self._get_copies()
        x_tmp = np.full((*original_shape, len(self.features)), fill_value=np.nan)
        x_tmp[mask] = x
        x = x_tmp
        del x_tmp
        if y is not None:
            y_tmp = np.full((*original_shape, len(self.targets)), fill_value=np.nan)
            y_tmp[mask] = y
            y = y_tmp
            del y_tmp
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

    def dataset_from_predictions(
        self,
        preds: np.ndarray,
        ensemble_axis: Optional[int] = None,
        targets: Optional[Sequence[Hashable]] = None,
    ) -> xr.Dataset:
        if not self._is_stacked:
            raise ValueError("Dataset should be stacked first.")
        if self.targets is None and targets is None:
            raise ValueError("Please specify argument 'targets'")
        else:
            targets = targets or self.targets
        event_shape = [
            len(c) for dim, c in self.coords.items() if dim not in self.batch_dims
        ]
        full_shape = [self.x.shape[0], *event_shape, len(targets)]
        dims = list(self.dims)
        coords = self.coords | {"v": targets}
        if ensemble_axis is not None:
            full_shape.insert(ensemble_axis, preds.shape[ensemble_axis])
            dims.insert(ensemble_axis, "realization")
            coords = coords | {"realization": np.arange(preds.shape[ensemble_axis])}
        if self.mask is not None:
            out = np.full(full_shape, fill_value=np.nan)
            out = xr.Variable(dims, out)
            out[{"s": self.mask}] = preds
        else:
            out = xr.Variable(dims, preds)
        out = out.unstack(s={dim: len(self.coords[dim]) for dim in self.batch_dims})
        out = xr.DataArray(out, coords=coords)
        return out.to_dataset("v")

    def _get_copies(self):
        x = self.x.copy()
        y = self.y.copy() if self.y is not None else None
        w = self.w.copy() if self.w is not None else None
        del self.x, self.y, self.w
        return x, y, w

    def __repr__(self) -> str:
        x, y, w = self._as_variables()
        ysize = dict(y.sizes) if y is not None else None
        wsize = dict(w.sizes) if w is not None else None
        out = f"Dataset(x={dict(x.sizes)}, y={ysize}, w={wsize})"
        return out


class DataLoader(tf.keras.utils.Sequence):
    """A dataloader for mlpp.

    Parameters
    ----------
    dataset: `Dataset`
        The `Dataset` object.
    batch_size: integer
        Size of each batch.
    shuffle: bool
        Enable or disable shuffling before each iteration.
    block_size: int
        Use to define the size of sample blocks, so that during shuffling indices within
        each block stay in their original order, but the blocks themselves are shuffled.
        Default value is 1, which is equivalent to a normal shuffling. This parameter
        is ignored if `shuffle=False`.


    Example:
    >>> train_dataloader = DataLoader(
    ...    train_dataset,
    ...    batch_size=2056,
    ...    shuffle=True,
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
        block_size: int = 1,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.block_size = block_size
        self.num_samples = len(self.dataset.x)
        self.num_batches = (
            self.num_samples // batch_size if batch_size <= self.num_samples else 1
        )
        self._indices = tf.range(self.num_samples)
        self._seed = 0
        self._reset()

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, index) -> tuple[tf.Tensor, ...]:
        if index >= self.num_batches:
            self._reset()
            raise IndexError
        start = index * self.batch_size
        end = index * self.batch_size + self.batch_size
        output = [self.dataset.x[start:end], self.dataset.y[start:end]]
        if self.dataset.w is not None:
            output.append(self.dataset.w[start:end])
        return tuple(output)

    def on_epoch_end(self) -> None:
        self._reset()

    def _shuffle_indices(self) -> None:
        """
        Shuffle the batch indices, with the option to define blocks, so that indices within
        each block stay in their original order, but the blocks themselves are shuffled.
        """
        if self.block_size == 1:
            self._indices = tf.random.shuffle(self._indices, seed=self._seed)
            return
        num_blocks = self._indices.shape[0] // self.block_size
        reshaped_indices = tf.reshape(
            self._indices[: num_blocks * self.block_size], (num_blocks, self.block_size)
        )
        shuffled_blocks = tf.random.shuffle(reshaped_indices, seed=self._seed)
        shuffled_indices = tf.reshape(shuffled_blocks, [-1])
        # Append any remaining elements if the number of indices isn't a multiple of the block size
        if shuffled_indices.shape[0] % self.block_size:
            remainder = self._indices[num_blocks * self.block_size :]
            shuffled_indices = tf.concat([shuffled_indices, remainder], axis=0)
        self._indices = shuffled_indices

    def _reset(self) -> None:
        """Reset iterator and shuffles data if needed"""
        self.index = 0
        if self.shuffle:
            self._shuffle_indices()
            self.dataset.x = tf.gather(self.dataset.x, self._indices)
            self.dataset.y = tf.gather(self.dataset.y, self._indices)
            if self.dataset.w is not None:
                self.dataset.w = tf.gather(self.dataset.w, self._indices)
            self._seed += 1

    def _to_device(self, device) -> None:
        """Transfer data to a device"""
        with tf.device(device):
            self.dataset.x = tf.constant(self.dataset.x)
            self.dataset.y = tf.constant(self.dataset.y)
            if self.dataset.w is not None:
                self.dataset.w = tf.constant(self.dataset.w)


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

    def apply(
        self, x: xr.Dataset, y: xr.Dataset, w: Optional[xr.Dataset] = None
    ) -> tuple[xr.Dataset, ...]:
        """Apply the provided filters to the input datasets."""

        if self.qa_mask is not None:
            y = y.where(~self.qa_mask)

        if self.x_filter is not None:
            x = x.assign_coords(is_valid=self.x_filter(x))

        out = (x, y) if w is None else (x, y, w)

        return out
