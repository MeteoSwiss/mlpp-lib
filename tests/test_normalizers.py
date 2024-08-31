import numpy as np
import pytest
import xarray as xr

from mlpp_lib.normalizers import DataTransformer


def get_class_attributes(cls):
    class_attrs = {
        name: field.default for name, field in cls.__dataclass_fields__.items()
    }
    return class_attrs


def test_fit(datatransformations, data_transformer, features_multi):

    for i, datatransform in enumerate(datatransformations):
        datatransform.fit(features_multi, variables=[f"var{i}"])

    data_transformer.fit(features_multi)

    assert all(
        (
            np.allclose(
                getattr(datatransform, attr),
                getattr(data_transformer.parameters[i][0], attr),
                equal_nan=True,
            )
            for attr in get_class_attributes(datatransform)
        )
        for datatransform in datatransformations
    )


def test_transform(datatransformations, data_transformer, features_multi):

    features_individual = features_multi.copy()
    for i, datatransform in enumerate(datatransformations):
        datatransform.fit(features_multi, variables=[f"var{i}"])
        features_individual = datatransform.transform(
            features_individual, variables=[f"var{i}"]
        )[0]

    data_transformer.fit(features_multi)
    features_multi = data_transformer.transform(features_multi)[0]

    assert all(
        np.allclose(
            features_individual[f"var{i}"].values,
            features_multi[f"var{i}"].values,
            equal_nan=True,
        )
        for i in range(len(datatransformations))
    )


def test_inverse_transform(datatransformations, data_transformer, features_multi):

    original_data = features_multi.copy().astype("float32")
    features_individual = features_multi.copy()
    for i, datatransform in enumerate(datatransformations):
        datatransform.fit(features_multi, variables=[f"var{i}"])
        features_individual = datatransform.transform(
            features_individual, variables=[f"var{i}"]
        )[0]
    inv_ds_individual = features_individual.copy()
    for i, datatransform in enumerate(datatransformations):
        inv_ds_individual = datatransform.inverse_transform(
            inv_ds_individual, variables=[f"var{i}"]
        )[0]

    data_transformer.fit(features_multi)
    ds_multi = data_transformer.transform(features_multi)[0]
    inv_ds_multi = data_transformer.inverse_transform(ds_multi)[0]

    assert all(
        np.allclose(
            inv_ds_individual[f"var{i}"].values,
            inv_ds_multi[f"var{i}"].values,
            equal_nan=True,
        )
        for i in range(len(datatransformations))
    ), "Inverse transform is not equal between individual data transformations and data_transformer"

    assert all(
        np.allclose(
            original_data[f"var{i}"].values,
            inv_ds_individual[f"var{i}"].values,
            equal_nan=True,
            atol=1e-6,
        )
        for i in range(len(datatransformations))
    ), "Inverse transform is not equal between transformed individual data transformations and original features"

    assert all(
        np.allclose(
            original_data[f"var{i}"].values,
            inv_ds_multi[f"var{i}"].values,
            equal_nan=True,
            atol=1e-6,
        )
        for i in range(len(datatransformations))
    ), "Inverse transform is not equal between transformed data_transformer and original features"


def test_serialization(data_transformer, features_multi, tmp_path):

    fn_multi = f"{tmp_path}/data_transformer.json"

    data_transformer.fit(features_multi)
    data_transformer.save_json(fn_multi)
    new_datatransformer = DataTransformer.from_json(fn_multi)

    assert data_transformer.method_vars_dict == new_datatransformer.method_vars_dict
    assert data_transformer.default == new_datatransformer.default
    assert data_transformer.fillvalue == new_datatransformer.fillvalue
    assert data_transformer.transformers == new_datatransformer.transformers


class TestLegacyStandardizer:
    @pytest.fixture
    def standardizer(self):
        from mlpp_lib.standardizers import Standardizer

        return Standardizer(fillvalue=-5)

    def test_fit(self, standardizer, features_dataset):
        standardizer.fit(features_dataset)
        assert all(
            var in standardizer.mean.data_vars for var in features_dataset.data_vars
        )
        assert all(
            var in standardizer.std.data_vars for var in features_dataset.data_vars
        )
        assert standardizer.fillvalue == -5

    def test_transform(self, standardizer, features_dataset):
        standardizer.fit(features_dataset)
        ds = standardizer.transform(features_dataset)[0]
        assert all(var in ds.data_vars for var in features_dataset.data_vars)
        assert all(np.isclose(ds[var].mean().values, 0) for var in ds.data_vars)
        assert all(np.isclose(ds[var].std().values, 1) for var in ds.data_vars)

    def test_inverse_transform(self, standardizer, features_dataset):
        standardizer.fit(features_dataset)
        ds = standardizer.transform(features_dataset)[0]
        inv_ds = standardizer.inverse_transform(ds)[0]

        assert all(
            np.allclose(
                inv_ds[var].values,
                features_dataset[var].values,
                equal_nan=True,
                atol=1e-6,
            )
            for var in features_dataset.data_vars
        )
        assert all(var in inv_ds.data_vars for var in features_dataset.data_vars)

    def test_retro_compatibility(self, standardizer, features_multi):
        standardizer.fit(features_multi)
        dict_stand = standardizer.to_dict()
        data_transformer = DataTransformer.from_dict(dict_stand)

        assert all(
            [
                np.allclose(
                    getattr(data_transformer.transformers["Standardizer"][0], attr)[
                        var
                    ].values,
                    getattr(standardizer, attr)[var].values,
                    equal_nan=True,
                )
                for var in getattr(standardizer, attr).data_vars
            ]
            if isinstance(getattr(standardizer, attr), xr.Dataset)
            else np.allclose(
                getattr(data_transformer.transformers["Standardizer"][0], attr),
                getattr(standardizer, attr),
            )
            for attr in get_class_attributes(standardizer)
        )
