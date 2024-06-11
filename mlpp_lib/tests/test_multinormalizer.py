import xarray as xr
import numpy as np
import json
import logging
import mlpp_lib.standardizers as st

LOGGER = logging.getLogger(__name__)

def setup_logger(log_file, level="INFO", format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"):
    logging.basicConfig(
        level=logging.getLevelName(level),
        format=format,
        datefmt=datefmt,
        filename=log_file,
        filemode="w",
    )


def get_class_attributes(cls):
    class_attrs = {name: field.default for name, field in cls.__dataclass_fields__.items()}
    return class_attrs

def check_equality(ds1, ds2):
    # Convert all data variables to a common type

    if not isinstance(ds1, xr.Dataset) or not isinstance(ds2, xr.Dataset):
        if ds1 == ds2: 
            return True
        else:
            LOGGER.info(f"Values are not equal: {ds1} != {ds2}")
        return False

    ds1_converted = ds1
    ds2_converted = ds2
    
    try:
        # Use xarray's testing function to check for equality with numerical tolerance
        xr.testing.assert_allclose(ds1_converted, ds2_converted)
        return True
    except AssertionError as e:
        # If there are differences, catch the exception and print the differences
        LOGGER.info("Datasets are not equal. Differences:")
        LOGGER.info(e)
        # Optionally, print a detailed difference
        LOGGER.info("\nDetailed differences:")
        ds1_vars = set(ds1.data_vars)
        ds2_vars = set(ds2.data_vars)
        
        # Check for differences in variables
        for var in ds1_vars.union(ds2_vars):
            if var not in ds1_vars:
                LOGGER.info(f"Variable {var} is missing in the first dataset.")
            elif var not in ds2_vars:
                LOGGER.info(f"Variable {var} is missing in the second dataset.")
            else:
                array1 = ds1_converted[var].values
                array2 = ds2_converted[var].values
                if not np.allclose(array1, array2):
                    diff = array1 - array2
                    LOGGER.info(f"Differences in variable '{var}':")
                    LOGGER.info(diff)
        return False


def create_dummy_dataset(nb_var=2):
    data = xr.Dataset(
        {
            f"var{i}": (("x", "y"), np.arange(4**i, 4**i+4).reshape(2, 2))\
            for i in range(nb_var)
        },
        coords={"x": [10, 20], "y": [1, 2]},
    )
    return data


def test_fit_data(normalizers, multinormalizer, data):

    for i, norma in enumerate(normalizers):
        norma.fit(data, variables=[f"var{i}"])

    multinormalizer.fit(data)

    err = False

    for i, normalizer in enumerate(normalizers):
        for attr in get_class_attributes(normalizer):
            if check_equality(getattr(normalizer, attr), getattr(multinormalizer.method_vars_list[i][0], attr)):
                LOGGER.info(f"\u2705 Attribute {attr} is equal between individual normalizer and multinormalizer")
            else:
                LOGGER.error(f"\u274c Attribute {attr} is not equal between individual normalizer and multinormalizer")
                err = True

    if err:
        raise ValueError("Attributes are not equal between individual normalizer and multinormalizer")

    return normalizers, multinormalizer


def test_transform_data(normalizers, multinormalizer, data):

    data_individual = data.copy()
    data_multi = data.copy()
    for i, norma in enumerate(normalizers):

        data_individual = norma.transform(data_individual, variables=[f"var{i}"])[0]

    data_multi = multinormalizer.transform(data_multi)[0]

    err = False

    if check_equality(data_individual, data_multi):
        LOGGER.info("\u2705 Data is equal after transformation")
    else:
        LOGGER.error("\u274c Data is not equal after transformation")
        err = True

    if err:
        raise ValueError("Data is not equal after transformation")

    return data_individual, data_multi


def test_inverse_transform_data(normalizers, multinormalizer, data_individual, data_multi, data):

    for i, norma in enumerate(normalizers):
        data_individual = norma.inverse_transform(data_individual, variables=[f"var{i}"])[0]

    data_multi = multinormalizer.inverse_transform(data_multi)[0]

    err = False

    if check_equality(data_individual, data_multi):
        LOGGER.info("\u2705 Data is equal after inverse transformation")
    else:
        LOGGER.error("\u274c Data is not equal after inverse transformation")
        err = True

    if check_equality(data_individual, data):
        LOGGER.info("\u2705 Inverse_tranform(transform(data)) == data")
    else:
        LOGGER.error("\u274c Inverse_tranform(transform(data)) != data")
        err = True

    if err:
        raise ValueError("Data is not equal after inverse transformation")

    return data_individual, data_multi


def test_save_as_dict(normalizers, multinormalizer):

    normalizers_dicts = {}
    for i, normalizer in enumerate(normalizers):
        norma_dict = normalizer.to_dict()
        norma_dict["channels"] = [f"var{i}"]

        normalizers_dicts[f"{normalizer.name}"] = norma_dict

    multi_dict = multinormalizer.to_dict()

    if check_equality(normalizers_dicts, multi_dict):
        LOGGER.info("\u2705 Normalizers are equal after saving as dict")
    else:
        LOGGER.error("\u274c Normalizers are not equal after saving as dict")
        raise ValueError("Normalizers are not equal after saving as dict")
    
    return normalizers_dicts, multi_dict


def test_load_from_dict(normalizers, multinormalizer, normalizers_dicts, multi_dict):

    normalizers_loaded = []
    for i, normalizer in enumerate(normalizers):
        normalizer_load = st.create_instance_from_str(normalizer.name).from_dict(normalizers_dicts[f"{normalizer.name}"])
        normalizers_loaded.append(normalizer_load)

    multi_loaded = st.create_instance_from_dict(multi_dict)

    err = False

    for i, norma_loaded in enumerate(normalizers_loaded):
        for attr in get_class_attributes(norma_loaded):
            if check_equality(getattr(norma_loaded, attr), getattr(multi_loaded.method_vars_list[i][0], attr)):
                LOGGER.info(f"\u2705 Attribute {attr} is equal between individual normalizer and multinormalizer")
            else:
                LOGGER.error(f"\u274c Attribute {attr} is not equal between individual normalizer and multinormalizer")
                err = True

    if err:
        raise ValueError("Attributes are not equal between individual normalizer and multinormalizer")

    return normalizers_loaded, multi_loaded


def test_main(normalizer_list):

    data = create_dummy_dataset()
    normalizer_individual = []
    method_var_dict = {normalizer: [f"var{i}"] for i, normalizer in enumerate(normalizer_list)}
    LOGGER.info(f"Method var dict: {method_var_dict}")
    multinormalizer = st.MultiNormalizer(method_var_dict=method_var_dict)
    
    for normalizer in normalizer_list:
        normalizer_individual.append(st.create_instance_from_str(normalizer))

    normalizers, multinormalizer = test_fit_data(normalizer_individual, multinormalizer, data)

    data_individual, data_multi = test_transform_data(normalizers, multinormalizer, data)
    LOGGER.info(data_individual)

    data_individual, data_multi = test_inverse_transform_data(normalizers, multinormalizer, data_individual, data_multi, data)

    normalizers_dicts, multi_dict = test_save_as_dict(normalizers, multinormalizer)

    normalizers_loaded, multi_loaded = test_load_from_dict(normalizers, multinormalizer, normalizers_dicts, multi_dict)


if __name__ == "__main__":
    setup_logger("tests_multi.log")

    normalizers = [n.name for n in st.Normalizer.__subclasses__() if not n.name == "MultiNormalizer"]
    LOGGER.info(f"Normalizers: {normalizers}")
    test_main(normalizer_list=normalizers)