from mlpp_lib.standardizers import MultiNormalizer, Standardizer, get_class_attributes
import numpy as np
import xarray as xr

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

    # not sure why we need that, np.allclose should set nans equal ...
    inv_ds_multi = xr.where(xr.apply_ufunc(np.isnan, inv_ds_multi), -5, inv_ds_multi)
    inv_ds_individual = xr.where(xr.apply_ufunc(np.isnan, inv_ds_individual), -5, inv_ds_individual)
    original_data = xr.where(xr.apply_ufunc(np.isnan, original_data), -5, original_data)

    assert all(
        np.allclose(inv_ds_individual[f"var{i}"].values, inv_ds_multi[f"var{i}"].values, equal_nan=True)
        for i in range(len(normalizers))
    ), "Inverse transform is not equal between individual normalizers and multinormalizer"

    """SHAPE = original_data["var0"].values.shape
    break_ = False
    for i in range(5):
        print(normalizers[i].name)
        for a in range(SHAPE[0]):
            for b in range(SHAPE[1]):
                for c in range(SHAPE[2]):
                    lhs = np.abs(original_data[f"var{i}"].values[a, b, c] - inv_ds_individual[f"var{i}"].values[a, b, c])
                    rhs = 1e-6 + 1e-5 * np.abs(inv_ds_individual[f"var{i}"].values[a, b, c])
                    if lhs > rhs:
                        print(a, b, c)
                        print(original_data[f"var{i}"].values[a, b, c], inv_ds_individual[f"var{i}"].values[a, b, c])
                        break_ = True
                    if break_:
                        break
                if break_:
                    break
            if break_:
                break
        break_ = False"""

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
    new_multinormalizer = MultiNormalizer.from_json(fn_multi)

    assert all(
        np.allclose(getattr(multinormalizer, attr), getattr(new_multinormalizer, attr), equal_nan=True)
        for attr in get_class_attributes(multinormalizer)
    )


def test_retro_compatibility(features_multi, tmp_path):

    fn_multi = f"{tmp_path}/multinormalizer.json"

    standardizer = Standardizer()
    standardizer.fit(features_multi)
    standardizer.save_json(fn_multi)
    multinormalizer = MultiNormalizer.from_json(fn_multi)
    
    assert all(
        [np.allclose(getattr(multinormalizer.parameters[0][0], attr)[var].values, getattr(standardizer, attr)[var].values, equal_nan=True)
         for var in getattr(standardizer, attr).data_vars] if type(getattr(standardizer, attr))==xr.Dataset
        else np.allclose(getattr(multinormalizer.parameters[0][0], attr), getattr(standardizer, attr))
        for attr in get_class_attributes(standardizer)
    )