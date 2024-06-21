import os
from pathlib import Path

import pytest
import xarray as xr
import numpy as np

from mlpp_lib.datasets import Dataset, DataModule
from mlpp_lib.model_selection import DataSplitter
from mlpp_lib.standardizers import DataTransformer
from .test_model_selection import ValidDataSplitterOptions

ZARR_MISSING = "zarr" not in xr.backends.list_engines()


class TestDataModule:

    features = ["coe:x1", "obs:x3", "dem:x4"]
    targets = ["obs:y1"]
    batch_dims = ["forecast_reference_time", "t", "station"]

    splitter_options = ValidDataSplitterOptions(time="lists", station="lists")
    splitter = DataSplitter(splitter_options.time_split, splitter_options.station_split)

    @pytest.fixture
    def data_transformer(self, features_dataset):
        data_transformer = DataTransformer()
        data_transformer.fit(features_dataset)
        return data_transformer

    @pytest.fixture  # https://docs.pytest.org/en/6.2.x/tmpdir.html
    def write_datasets_zarr(self, tmp_path, features_dataset, targets_dataset):
        features_dataset.to_zarr(tmp_path / "features.zarr", mode="w")
        targets_dataset.to_zarr(tmp_path / "targets.zarr", mode="w")

    @pytest.mark.skipif(ZARR_MISSING, reason="missing zarr")
    @pytest.mark.usefixtures("write_datasets_zarr")
    def test_setup_fit_default_fromfile(self, tmp_path: Path):
        dm = DataModule(
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
            data_dir=tmp_path.as_posix() + "/",
        )
        dm.setup("fit")

    def test_setup_fit_default_fromds(self, features_dataset, targets_dataset):
        dm = DataModule(
            features_dataset,
            targets_dataset,
            self.batch_dims,
            self.splitter,
        )
        dm.setup("fit")

    @pytest.mark.skipif(ZARR_MISSING, reason="missing zarr")
    @pytest.mark.usefixtures("write_datasets_zarr")
    def test_setup_test_default_fromfile(self, tmp_path: Path, data_transformer):
        dm = DataModule(
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
            data_dir=tmp_path.as_posix() + "/",
            data_transformer=data_transformer,
        )
        dm.setup("test")

    @pytest.mark.skipif(ZARR_MISSING, reason="missing zarr")
    @pytest.mark.usefixtures("write_datasets_zarr")
    def test_setup_fit_thinning(self, tmp_path: Path):
        dm = DataModule(
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
            data_dir=tmp_path.as_posix() + "/",
            thinning={"forecast_reference_time": 2},
        )
        dm.setup("fit")

    @pytest.mark.skipif(ZARR_MISSING, reason="missing zarr")
    @pytest.mark.usefixtures("write_datasets_zarr")
    def test_setup_fit_weights(self, tmp_path: Path):
        dm = DataModule(
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
            data_dir=tmp_path.as_posix() + "/",
            sample_weighting=["coe:x1"],
        )
        dm.setup("fit")


class TestDataset:
    @pytest.fixture
    def dataset(self, features_dataset: xr.Dataset, targets_dataset: xr.Dataset):
        return Dataset.from_xarray_datasets(features_dataset, targets_dataset)

    @pytest.fixture
    def dataset_only_x(self, features_dataset: xr.Dataset):
        return Dataset.from_xarray_datasets(features_dataset)

    @pytest.fixture
    def coords(self, features_dataset):
        dims = list(features_dataset.dims)
        return {dim: features_dataset.coords[dim] for dim in dims}

    @pytest.fixture
    def dims(self, features_dataset):
        return list(features_dataset.dims)

    @pytest.fixture
    def features(self, features_dataset):
        return list(features_dataset.data_vars)

    @pytest.fixture
    def targets(self, targets_dataset):
        return list(targets_dataset.data_vars)

    def test_from_xarray_datasets(self, dataset, dims, coords, features, targets):
        assert dataset.x.shape == (
            *tuple(len(c) for c in coords.values()),
            len(features),
        )
        assert dataset.y.shape == (
            *tuple(len(c) for c in coords.values()),
            len(targets),
        )
        assert dataset.dims == [*dims, "v"]
        assert dataset.coords.keys() == coords.keys()
        assert [len(c) for c in dataset.coords] == [len(c) for c in coords]
        assert dataset.features == features
        assert dataset.targets == targets

    def test_from_xarray_datasets_only_x(self, dataset_only_x, dims, coords, features):
        assert dataset_only_x.x.shape == (
            *tuple(len(c) for c in coords.values()),
            len(features),
        )
        assert dataset_only_x.y is None
        assert dataset_only_x.dims == [*dims, "v"]
        assert dataset_only_x.coords.keys() == coords.keys()
        assert [len(c) for c in dataset_only_x.coords] == [len(c) for c in coords]
        assert dataset_only_x.features == features

    @pytest.mark.parametrize(
        "batch_dims",
        [
            ("forecast_reference_time", "t", "station"),
            ("forecast_reference_time", "station"),
        ],
        ids=lambda x: repr(x),
    )
    def test_stack(self, dataset, dims, coords, features, targets, batch_dims):
        ds = dataset.stack(batch_dims)
        event_dims = tuple(set(dims) - set(batch_dims))
        n_samples = np.prod([len(c) for d, c in coords.items() if d in batch_dims])
        assert ds.x.shape == (
            n_samples,
            *tuple(len(coords[d]) for d in event_dims),
            len(features),
        )
        assert ds.y.shape == (
            n_samples,
            *tuple(len(coords[d]) for d in event_dims),
            len(targets),
        )
        assert ds.dims == ["s", *event_dims, "v"]
        assert ds.coords.keys() == coords.keys()
        assert [len(c) for c in ds.coords] == [len(c) for c in coords]

    @pytest.mark.parametrize(
        "batch_dims",
        [
            ("forecast_reference_time", "t", "station"),
            ("forecast_reference_time", "station"),
        ],
        ids=lambda x: repr(x),
    )
    def test_stack_only_x(self, dataset_only_x, dims, coords, features, batch_dims):
        ds = dataset_only_x.stack(batch_dims)
        event_dims = tuple(set(dims) - set(batch_dims))
        n_samples = np.prod([len(c) for d, c in coords.items() if d in batch_dims])
        assert ds.x.shape == (
            n_samples,
            *tuple(len(coords[d]) for d in event_dims),
            len(features),
        )
        assert ds.y is None
        assert ds.dims == ["s", *event_dims, "v"]
        assert ds.coords.keys() == coords.keys()
        assert [len(c) for c in ds.coords] == [len(c) for c in coords]

    @pytest.mark.parametrize("drop_nans", [True, False], ids=lambda x: f"drop_nans={x}")
    def test_unstack(self, dataset, dims, coords, features, targets, drop_nans):
        batch_dims = ("forecast_reference_time", "t", "station")
        if drop_nans:
            ds = dataset.stack(batch_dims).drop_nans()
        else:
            ds = dataset.stack(batch_dims)

        ds = ds.unstack()
        assert ds.x.shape == (*tuple(len(c) for c in coords.values()), len(features))
        assert ds.y.shape == (*tuple(len(c) for c in coords.values()), len(targets))
        assert ds.dims == (*dims, "v")
        assert ds.coords.keys() == coords.keys()
        assert [len(c) for c in ds.coords] == [len(c) for c in coords]
        assert ds.features == features
        assert ds.targets == targets

    @pytest.mark.parametrize("drop_nans", [True, False], ids=lambda x: f"drop_nans={x}")
    def test_unstack_only_x(self, dataset_only_x, dims, coords, features, drop_nans):
        batch_dims = ("forecast_reference_time", "t", "station")
        if drop_nans:
            ds = dataset_only_x.stack(batch_dims).drop_nans()
        else:
            ds = dataset_only_x.stack(batch_dims)

        ds = ds.unstack()
        assert ds.x.shape == (*tuple(len(c) for c in coords.values()), len(features))
        assert ds.y is None
        assert ds.dims == (*dims, "v")
        assert ds.coords.keys() == coords.keys()
        assert [len(c) for c in ds.coords] == [len(c) for c in coords]
        assert ds.features == features

    def test_drop_nans(self, dataset, dims, coords, features, targets):
        batch_dims = ("forecast_reference_time", "t", "station")
        ds = dataset.stack(batch_dims).drop_nans()
        event_dims = tuple(set(dims) - set(batch_dims))
        n_samples = np.prod([len(c) for d, c in coords.items() if d in batch_dims])
        assert ds.x.shape == (
            ds.mask.sum(),
            *tuple(len(coords[d]) for d in event_dims),
            len(features),
        )
        assert ds.y.shape == (
            ds.mask.sum(),
            *tuple(len(coords[d]) for d in event_dims),
            len(targets),
        )
        assert len(ds.mask) == n_samples
        assert ds.dims == ["s", *event_dims, "v"]
        assert ds.coords.keys() == coords.keys()

    def test_drop_nans_only_x(self, dataset_only_x, dims, coords, features):
        batch_dims = ("forecast_reference_time", "t", "station")
        ds = dataset_only_x.stack(batch_dims).drop_nans()
        event_dims = tuple(set(dims) - set(batch_dims))
        n_samples = np.prod([len(c) for d, c in coords.items() if d in batch_dims])
        assert ds.x.shape == (
            ds.mask.sum(),
            *tuple(len(coords[d]) for d in event_dims),
            len(features),
        )
        assert ds.y is None
        assert len(ds.mask) == n_samples
        assert ds.dims == ["s", *event_dims, "v"]
        assert ds.coords.keys() == coords.keys()

    @pytest.mark.parametrize(
        "batch_dims",
        [
            ("forecast_reference_time", "t", "station"),
            ("forecast_reference_time", "station"),
        ],
        ids=lambda x: repr(x),
    )
    def test_dataset_from_predictions(self, dataset, batch_dims):
        n_samples = 3
        ds = dataset.stack(batch_dims)
        ds = ds.drop_nans()
        predictions = np.random.randn(n_samples, *ds.y.shape)
        ds_pred = ds.dataset_from_predictions(predictions, ensemble_axis=0)
        assert isinstance(ds_pred, xr.Dataset)
        assert ds_pred.dims["realization"] == n_samples
        assert all([ds_pred.dims[c] == ds.coords[c].size for c in ds.coords])
        assert list(ds_pred.data_vars) == ds.targets

    @pytest.mark.parametrize(
        "batch_dims",
        [
            ("forecast_reference_time", "t", "station"),
            ("forecast_reference_time", "station"),
        ],
        ids=lambda x: repr(x),
    )
    def test_dataset_from_predictions_only_x(self, dataset_only_x, batch_dims):
        n_samples = 3
        targets = ["obs:y1", "obs:y2"]
        ds = dataset_only_x.stack(batch_dims)
        # Note that here we do not drop nan, hence the mask is not created!
        predictions = np.random.randn(n_samples, *ds.x.shape[:-1], len(targets))
        ds_pred = ds.dataset_from_predictions(
            predictions, ensemble_axis=0, targets=targets
        )
        assert isinstance(ds_pred, xr.Dataset)
        assert ds_pred.dims["realization"] == n_samples
        assert all([ds_pred.dims[c] == ds.coords[c].size for c in ds.coords])
        assert list(ds_pred.data_vars) == targets
