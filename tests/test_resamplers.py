import pytest

import numpy as np

from mlpp_lib.datasets import Dataset
from mlpp_lib.resamplers import RegressionResampler


@pytest.fixture
def make_dataset(features_dataset, targets_dataset):
    ds = Dataset.from_xarray_datasets(features_dataset, targets_dataset[["obs:y1"]])
    return ds.stack(["forecast_reference_time", "t", "station"])


def test_fit_returns_valid_probabilities(make_dataset):
    ds = make_dataset
    x, y = ds.x, ds.y
    resampler = RegressionResampler.fit(y, n_bins=None)
    prob = resampler.prob

    assert len(prob) == len(y)
    np.testing.assert_allclose(prob.sum(), 1.0)
    assert np.all(prob > 0)


def test_sample_indices_length_and_range(make_dataset):
    ds = make_dataset
    x, y = ds.x, ds.y
    resampler = RegressionResampler.fit(y)
    indices = resampler.sample_indices(size=9, random_seed=0)

    assert len(indices) == 9
    assert np.all((indices >= 0) & (indices < len(y)))


def test_resample_output_shapes(make_dataset):
    ds = make_dataset
    x, y = ds.x, ds.y
    resampler = RegressionResampler.fit(y)
    x_resampled, y_resampled = resampler.resample(x, y, size=9, random_seed=0)

    assert x_resampled.shape == (9, 4)
    assert y_resampled.shape == (9, 1)
    assert x_resampled.ndim == x.ndim
    assert y_resampled.ndim == y.ndim
    assert set(list(np.unique(y_resampled))).issubset(set(list(np.unique(y))))


def test_fit_resample_full_pipeline(make_dataset):
    ds = make_dataset
    x, y = ds.x, ds.y
    x_resampled, y_resampled = RegressionResampler.fit_resample(
        x, y=y, size=9, random_seed=0
    )

    assert x_resampled.shape == (9, 4)
    assert y_resampled.shape == (9, 1)
    assert set(list(np.unique(y_resampled))).issubset(set(list(np.unique(y))))


def test_resample_multiple_arrays(make_dataset):
    ds = make_dataset
    x, y = ds.x, ds.y
    z = y * 10  # Additional input array
    resampler = RegressionResampler.fit(y)
    x_resampled, z_resampled, y_resampled = resampler.resample(
        x, z, y, size=9, random_seed=0
    )

    assert x_resampled.shape == (9, 4)
    assert z_resampled.shape == (9, 1)
    assert y_resampled.shape == (9, 1)
    np.testing.assert_array_equal(y_resampled * 10, z_resampled)


def test_invalid_array_length_raises(make_dataset):
    ds = make_dataset
    x, y = ds.x, ds.y
    x_bad = x[:-1]  # Shorter than y
    resampler = RegressionResampler.fit(y)

    with pytest.raises(AssertionError):
        resampler.resample(x_bad, y)
