import numpy as np

from mlpp_lib.standardizers import Standardizer


def test_fit(features_dataset):
    standardizer = Standardizer(fillvalue=-5)
    standardizer.fit(features_dataset)
    assert all(var in standardizer.mean.data_vars for var in features_dataset.data_vars)
    assert all(var in standardizer.std.data_vars for var in features_dataset.data_vars)
    assert standardizer.fillvalue == -5


def test_transform(features_dataset):
    standardizer = Standardizer(fillvalue=-5)
    standardizer.fit(features_dataset)
    ds = standardizer.transform(features_dataset)[0]
    assert all(var in ds.data_vars for var in features_dataset.data_vars)
    assert all(np.isclose(ds[var].mean().values, 0) for var in ds.data_vars)
    assert all(np.isclose(ds[var].std().values, 1) for var in ds.data_vars)


def test_inverse_transform(features_dataset):
    standardizer = Standardizer(fillvalue=-5)
    standardizer.fit(features_dataset)
    ds = standardizer.transform(features_dataset)[0]
    inv_ds = standardizer.inverse_transform(ds)[0]

    assert all(
        np.allclose(inv_ds[var].values, features_dataset[var].values, equal_nan=True, atol=1e-6)
        for var in features_dataset.data_vars
    )
    assert all(var in inv_ds.data_vars for var in features_dataset.data_vars)


def test_serialization(features_dataset, tmp_path):
    fn = f"{tmp_path}/standardizer.json"
    standardizer = Standardizer(fillvalue=-5)
    standardizer.fit(features_dataset)
    standardizer.save_json(fn)
    new_standardizer = Standardizer.from_json(fn)
    assert standardizer.fillvalue == new_standardizer.fillvalue
    assert standardizer.mean.identical(new_standardizer.mean)
    assert standardizer.std.identical(new_standardizer.std)
