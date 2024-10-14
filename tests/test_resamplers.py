import pytest

import numpy as np

from mlpp_lib.datasets import Dataset
from mlpp_lib.resamplers import RegressionResampler


@pytest.fixture
def make_dataset(features_dataset, targets_dataset):
    ds = Dataset.from_xarray_datasets(features_dataset, targets_dataset[["obs:y1"]])
    return ds.stack(["forecast_reference_time", "t", "station"])


def test_regression_resampler_bins(make_dataset):

    ds = make_dataset

    resampled_data = RegressionResampler.fit_resample(ds.x, ds.y, n_bins=5)

    assert resampled_data[0].shape[0] == ds.x.shape[0]
    assert resampled_data[1].shape[0] == ds.y.shape[0]
    assert resampled_data[0].ndim == ds.x.ndim
    assert resampled_data[1].ndim == ds.y.ndim


def test_regression_resampler_labels(make_dataset):

    ds = make_dataset
    labels = np.random.randint(0, 3, ds.y.size)
    resampler = RegressionResampler.fit(labels)
    resampled_data = resampler.resample(ds.x, ds.y)

    assert resampled_data[0].shape[0] == ds.x.shape[0]
    assert resampled_data[1].shape[0] == ds.y.shape[0]
    assert resampled_data[0].ndim == ds.x.ndim
    assert resampled_data[1].ndim == ds.y.ndim
