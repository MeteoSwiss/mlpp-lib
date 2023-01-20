import os
from pathlib import Path

import pytest
import xarray as xr
import numpy as np

from mlpp_lib.datasets import Dataset, DataModule
from mlpp_lib.model_selection import DataSplitter
from mlpp_lib.standardizers import Standardizer
from .test_model_selection import ValidDataSplitterOptions


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


    @pytest.mark.usefixtures("write_datasets_zarr")
    def test_setup_fit_default(self, tmp_path: Path):
        dm = DataModule(
            tmp_path.as_posix() + "/",
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
        )
        dm.setup("fit")

    @pytest.mark.usefixtures("write_datasets_zarr")
    def test_setup_test_default(self, tmp_path: Path, standardizer):
        dm = DataModule(
            tmp_path.as_posix() + "/",
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
            standardizer=standardizer,
        )
        dm.setup("test")

    @pytest.mark.usefixtures("write_datasets_zarr")
    def test_setup_fit_thinning(self, tmp_path: Path):
        dm = DataModule(
            tmp_path.as_posix() + "/",
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
            thinning={"forecast_reference_time": 2}
        )
        dm.setup("fit")

    @pytest.mark.usefixtures("write_datasets_zarr")
    def test_setup_fit_weights(self, tmp_path: Path):
        dm = DataModule(
            tmp_path.as_posix() + "/",
            self.features,
            self.targets,
            self.batch_dims,
            self.splitter,
            sample_weighting=["coe:x1"]
        )
        dm.setup("fit")

def test_get_tensor_dataset_fit(
    features_dataset, targets_dataset, get_dummy_keras_model
):
    """Test that output can be used to fit keras model"""
    tensors = get_tensor_dataset(features_dataset, targets_dataset, None, event_dims=[])
    model = get_dummy_keras_model(tensors[0].shape[1], tensors[1].shape[1])
    model.fit(
        x=tensors[0],
        y=tensors[1],
        sample_weight=tensors[2],
        epochs=1,
    )


def test_split_dataset(features_dataset):
    """"""
    splits = dict(
        a=dict(station=["AAA", "BBB"]),
        b=dict(t=slice(0, 1)),
        c=dict(t=slice(0, 1), station=["AAA", "BBB"]),
    )
    out = split_dataset(features_dataset, splits)
    dims_in = features_dataset.dims
    assert list(out.keys()) == ["a", "b", "c"]
    assert out["a"].dims == dict(dims_in, station=2)
    assert out["b"].dims == dict(dims_in, t=2)
    assert out["c"].dims == dict(dims_in, t=2, station=2)


def test_split_dataset_missing_dims(features_dataset):
    """"""
    splits = dict(
        a=dict(station=["AAA", "BBB"], asd=[1, 2, 3]),
    )
    with pytest.raises(KeyError):
        out = split_dataset(features_dataset, splits, ignore_missing_dims=False)
    out = split_dataset(features_dataset, splits, ignore_missing_dims=True)
    assert list(out.keys()) == ["a"]


@pytest.mark.parametrize("ignore_missing_dims,", (True, False))
def test_split_dataset_None(ignore_missing_dims):
    """"""
    splits = dict(
        a=dict(station=["AAA", "BBB"]),
        b=dict(t=slice(0, 1)),
        c=dict(t=slice(0, 1), station=["AAA", "BBB"]),
    )
    out = split_dataset(None, splits, ignore_missing_dims=ignore_missing_dims)
    assert list(out.keys()) == ["a", "b", "c"]
    assert list(out.values()) == [None, None, None]
