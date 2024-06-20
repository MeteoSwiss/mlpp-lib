from mlpp_lib.standardizers import Normalizer, Standardizer
import numpy as np
import xarray as xr


def get_class_attributes(cls):
    class_attrs = {name: field.default for name, field in cls.__dataclass_fields__.items()}
    return class_attrs

def test_fit(normalizers, multinormalizer, features_multi):

    for i, normalizer in enumerate(normalizers):
        normalizer.fit(features_multi, variables=[f"var{i}"])

    multinormalizer.fit(features_multi)
    
    assert all(
        (np.allclose(getattr(normalizer, attr), getattr(multinormalizer.parameters[i][0], attr), equal_nan=True)
        for attr in get_class_attributes(normalizer))
        for normalizer in normalizers
    )


def test_transform(normalizers, multinormalizer, features_multi):

    features_individual = features_multi.copy()
    for i, normalizer in enumerate(normalizers):
        normalizer.fit(features_multi, variables=[f"var{i}"])
        features_individual = normalizer.transform(features_individual, variables=[f"var{i}"])[0]

    multinormalizer.fit(features_multi)
    features_multi = multinormalizer.transform(features_multi)[0]

    assert all(
        np.allclose(features_individual[f"var{i}"].values, features_multi[f"var{i}"].values, equal_nan=True)
        for i in range(len(normalizers))
    )


def test_inverse_transform(normalizers, multinormalizer, features_multi):

    original_data = features_multi.copy().astype("float32")
    features_individual = features_multi.copy()
    for i, normalizer in enumerate(normalizers):
        normalizer.fit(features_multi, variables=[f"var{i}"])
        features_individual = normalizer.transform(features_individual, variables=[f"var{i}"])[0]
    inv_ds_individual = features_individual.copy()
    for i, normalizer in enumerate(normalizers):
        inv_ds_individual = normalizer.inverse_transform(inv_ds_individual, variables=[f"var{i}"])[0]

    multinormalizer.fit(features_multi)
    ds_multi = multinormalizer.transform(features_multi)[0]
    inv_ds_multi = multinormalizer.inverse_transform(ds_multi)[0]

    assert all(
        np.allclose(inv_ds_individual[f"var{i}"].values, inv_ds_multi[f"var{i}"].values, equal_nan=True)
        for i in range(len(normalizers))
    ), "Inverse transform is not equal between individual normalizers and multinormalizer"

    assert all(
        np.allclose(original_data[f"var{i}"].values, inv_ds_individual[f"var{i}"].values, equal_nan=True, atol=1e-6)
        for i in range(len(normalizers))
    ), "Inverse transform is not equal between transformed individual normalizers and original features"

    assert all(
        np.allclose(original_data[f"var{i}"].values, inv_ds_multi[f"var{i}"].values, equal_nan=True, atol=1e-6)
        for i in range(len(normalizers))
    ), "Inverse transform is not equal between transformed multinormalizer and original features"


def test_serialization(multinormalizer, features_multi, tmp_path):

    fn_multi = f"{tmp_path}/multinormalizer.json"

    multinormalizer.fit(features_multi)
    multinormalizer.save_json(fn_multi)
    new_multinormalizer = Normalizer.from_json(fn_multi)

    assert all(
        np.allclose(getattr(multinormalizer, attr), getattr(new_multinormalizer, attr), equal_nan=True)
        for attr in get_class_attributes(multinormalizer)
    )


def test_retro_compatibility(features_multi):

    standardizer = Standardizer()
    standardizer.fit(features_multi)
    dict_stand = standardizer.to_dict()
    multinormalizer = Normalizer.from_dict(dict_stand)
    
    assert all(
        [np.allclose(getattr(multinormalizer.parameters[0][0], attr)[var].values, getattr(standardizer, attr)[var].values, equal_nan=True)
         for var in getattr(standardizer, attr).data_vars] if type(getattr(standardizer, attr))==xr.Dataset
        else np.allclose(getattr(multinormalizer.parameters[0][0], attr), getattr(standardizer, attr))
        for attr in get_class_attributes(standardizer)
    )