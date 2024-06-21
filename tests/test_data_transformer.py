from mlpp_lib.standardizers import DataTransformer, Standardizer
import numpy as np
import xarray as xr


def get_class_attributes(cls):
    class_attrs = {name: field.default for name, field in cls.__dataclass_fields__.items()}
    return class_attrs

def test_fit(datatransformations, data_transformer, features_multi):

    for i, datatransform in enumerate(datatransformations):
        datatransform.fit(features_multi, variables=[f"var{i}"])

    data_transformer.fit(features_multi)
    
    assert all(
        (np.allclose(getattr(datatransform, attr), getattr(data_transformer.parameters[i][0], attr), equal_nan=True)
        for attr in get_class_attributes(datatransform))
        for datatransform in datatransformations
    )


def test_transform(datatransformations, data_transformer, features_multi):

    features_individual = features_multi.copy()
    for i, datatransform in enumerate(datatransformations):
        datatransform.fit(features_multi, variables=[f"var{i}"])
        features_individual = datatransform.transform(features_individual, variables=[f"var{i}"])[0]

    data_transformer.fit(features_multi)
    features_multi = data_transformer.transform(features_multi)[0]

    assert all(
        np.allclose(features_individual[f"var{i}"].values, features_multi[f"var{i}"].values, equal_nan=True)
        for i in range(len(datatransformations))
    )


def test_inverse_transform(datatransformations, data_transformer, features_multi):

    original_data = features_multi.copy().astype("float32")
    features_individual = features_multi.copy()
    for i, datatransform in enumerate(datatransformations):
        datatransform.fit(features_multi, variables=[f"var{i}"])
        features_individual = datatransform.transform(features_individual, variables=[f"var{i}"])[0]
    inv_ds_individual = features_individual.copy()
    for i, datatransform in enumerate(datatransformations):
        inv_ds_individual = datatransform.inverse_transform(inv_ds_individual, variables=[f"var{i}"])[0]

    data_transformer.fit(features_multi)
    ds_multi = data_transformer.transform(features_multi)[0]
    inv_ds_multi = data_transformer.inverse_transform(ds_multi)[0]

    assert all(
        np.allclose(inv_ds_individual[f"var{i}"].values, inv_ds_multi[f"var{i}"].values, equal_nan=True)
        for i in range(len(datatransformations))
    ), "Inverse transform is not equal between individual data transformations and data_transformer"

    assert all(
        np.allclose(original_data[f"var{i}"].values, inv_ds_individual[f"var{i}"].values, equal_nan=True, atol=1e-6)
        for i in range(len(datatransformations))
    ), "Inverse transform is not equal between transformed individual data transformations and original features"

    assert all(
        np.allclose(original_data[f"var{i}"].values, inv_ds_multi[f"var{i}"].values, equal_nan=True, atol=1e-6)
        for i in range(len(datatransformations))
    ), "Inverse transform is not equal between transformed data_transformer and original features"


def test_serialization(data_transformer, features_multi, tmp_path):

    fn_multi = f"{tmp_path}/data_transformer.json"

    data_transformer.fit(features_multi)
    data_transformer.save_json(fn_multi)
    new_datatransformer = DataTransformer.from_json(fn_multi)

    assert all(
        np.allclose(getattr(data_transformer, attr), getattr(new_datatransformer, attr), equal_nan=True)
        for attr in get_class_attributes(data_transformer)
    )


def test_retro_compatibility(features_multi):

    standardizer = Standardizer()
    standardizer.fit(features_multi)
    dict_stand = standardizer.to_dict()
    data_transformer = DataTransformer.from_dict(dict_stand)
    
    assert all(
        [np.allclose(getattr(data_transformer.parameters[0][0], attr)[var].values, getattr(standardizer, attr)[var].values, equal_nan=True)
         for var in getattr(standardizer, attr).data_vars] if type(getattr(standardizer, attr))==xr.Dataset
        else np.allclose(getattr(data_transformer.parameters[0][0], attr), getattr(standardizer, attr))
        for attr in get_class_attributes(standardizer)
    )