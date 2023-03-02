import os
from pathlib import Path

import pytest
import xarray as xr
import numpy as np

from mlpp_lib.datasets import Dataset, DataModule
from mlpp_lib.model_selection import DataSplitter
from mlpp_lib.standardizers import Standardizer
from .test_model_selection import ValidDataSplitterOptions

ZARR_MISSING = "zarr" not in xr.backends.list_engines()


class TestDataModule:

    features = ["coe:x1", "obs:x3", "dem:x4"]
    targets = ["obs:y1"]
    batch_dims = ["forecast_reference_time", "t", "station"]

    splitter_options = ValidDataSplitterOptions(time="lists", station="lists")
    splitter = DataSplitter(splitter_options.time_split, splitter_options.station_split)

    @pytest.fixture
    def standardizer(self, features_dataset):
        standardizer = Standardizer()
        standardizer.fit(features_dataset)
        return standardizer

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
    def test_setup_test_default_fromfile(self, tmp_path: Path, standardizer):
        dm = DataModule(
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
            data_dir=tmp_path.as_posix() + "/",
            standardizer=standardizer,
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
