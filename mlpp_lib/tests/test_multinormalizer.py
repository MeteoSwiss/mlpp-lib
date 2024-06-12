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
            f"var{i}": (("x", "y"), np.arange(4*i+1, 4*(i+1)+1).reshape(2, 2))\
            for i in range(nb_var)
        },
        coords={"x": [10, 20], "y": [1, 2]},
    )
    LOGGER.info(f"Data:\n{data}")
    return data


def test_fit_data(normalizers, multinormalizer, data):

    LOGGER.info("Testing fit()...")
    for i, norma in enumerate(normalizers):
        norma.fit(data, variables=[f"var{i}"])

    multinormalizer.fit(data)

    err = False

    for i, normalizer in enumerate(normalizers):
        for attr in get_class_attributes(normalizer):
            if check_equality(getattr(normalizer, attr), getattr(multinormalizer.parameters[i][0], attr)):
                LOGGER.info(f"\u2705 Attribute {attr} is equal between {normalizer.name} and multinormalizer.{multinormalizer.parameters[i][0].name}")
            else:
                LOGGER.error(f"\u274c Attribute {attr} is not equal between {normalizer.name} and multinormalizer.{multinormalizer.parameters[i][0].name}")
                err = True

    if err:
        raise ValueError("Attributes are not equal between (at least) one of the individual normalizer and the multinormalizer")
    LOGGER.info("Fitting data seems to work\n")

    return normalizers, multinormalizer


def test_transform_data(normalizers, multinormalizer, data):

    LOGGER.info("Testing transform()...")
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
    LOGGER.info("Transforming data seems to work\n")

    return data_individual, data_multi


def test_inverse_transform_data(normalizers, multinormalizer, data_individual, data_multi, data):

    LOGGER.info("Testing inverse_transform()...")
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
    LOGGER.info("Inverse transforming data seems to work\n")

    return data_individual, data_multi


def test_save_as_dict(normalizers, multinormalizer):

    LOGGER.info("Testing to_dict()...")
    normalizers_dicts = {}
    for i, normalizer in enumerate(normalizers):
        norma_dict = normalizer.to_dict()
        norma_dict["channels"] = [f"var{i}"]

        normalizers_dicts[f"{normalizer.name}"] = norma_dict

    multi_dict = multinormalizer.to_dict()

    if check_equality(normalizers_dicts, multi_dict):
        LOGGER.info("\u2705 Normalizers are equal after saving as dict\n")
    else:
        LOGGER.error("\u274c Normalizers are not equal after saving as dict")
        raise ValueError("Normalizers are not equal after saving as dict")
    
    return normalizers_dicts, multi_dict


def test_load_from_dict(normalizers, multinormalizer, normalizers_dicts, multi_dict):

    LOGGER.info("Testing from_dict()...")
    normalizers_loaded = []
    for i, normalizer in enumerate(normalizers):
        normalizer_load = st.create_normalizer_from_str(normalizer.name).from_dict(normalizers_dicts[f"{normalizer.name}"])
        normalizers_loaded.append(normalizer_load)

    multi_loaded = st.create_normalizer_from_str(multinormalizer.name).from_dict(multi_dict)

    err = False

    for i, norma_loaded in enumerate(normalizers_loaded):
        for attr in get_class_attributes(norma_loaded):
            if check_equality(getattr(norma_loaded, attr), getattr(multi_loaded.parameters[i][0], attr)):
                LOGGER.info(f"\u2705 Attribute {attr} is equal between {norma_loaded.name} and multinormalizer.{multi_loaded.parameters[i][0].name}")
            else:
                LOGGER.error(f"\u274c Attribute {attr} is not equal between {norma_loaded.name} and multinormalizer.{multi_loaded.parameters[i][0].name}")
                err = True

    if err:
        raise ValueError("Attributes are not equal between (at least) one of the individual normalizer and the multinormalizer")
    LOGGER.info("Loading from dict seems to work\n")

    return normalizers_loaded, multi_loaded


def test_save_to_json(normalizers, multinormalizer, file_path):

    LOGGER.info("Testing save_json()...")
    filepaths = [f"{file_path}_{normalizers[i].name}.json" for i in range(len(normalizers))] + [f"{file_path}_{multinormalizer.name}.json"]
    err = False
    for i, normalizer in enumerate(normalizers):
        try:
            normalizer.save_json(out_fn=filepaths[i])
        except Exception as e:
            LOGGER.error(f"\u274c Error saving {normalizer.name}: {e}")
            err = True

    try:
        multinormalizer.save_json(out_fn=filepaths[-1])
    except Exception as e:
        LOGGER.error(f"\u274c Error saving {multinormalizer.name}: {e}")
        err = True

    if err:
        raise ValueError("Error during saving as json")
    LOGGER.info("\u2705 Everything saved as json\n")

    return filepaths


def test_load_from_json(normalizers, multinormalizer, filepaths):

    LOGGER.info("Testing from_json()...")
    normalizers_loaded = []
    err = False
    for i, normalizer in enumerate(normalizers):
        try:
            normalizer_load = st.create_normalizer_from_str(normalizer.name).from_json(in_fn=filepaths[i])
            normalizers_loaded.append(normalizer_load)
        except Exception as e:
            LOGGER.error(f"\u274c Error loading {normalizer.name}: {e}")
            err = True

    try:
        multi_loaded = st.create_normalizer_from_str(multinormalizer.name).from_json(in_fn=filepaths[-1])
    except Exception as e:
        LOGGER.error(f"\u274c Error loading {multinormalizer.name}: {e}")
        err = True

    if err:
        raise ValueError("Error during loading from json")
    else:
        LOGGER.info("\u2705 Everything loaded from json")

    err = False

    for i, norma_loaded in enumerate(normalizers_loaded):
        for attr in get_class_attributes(norma_loaded):
            if check_equality(getattr(norma_loaded, attr), getattr(multi_loaded.parameters[i][0], attr)):
                LOGGER.info(f"\u2705 Attribute {attr} is equal between {norma_loaded.name} and multinormalizer.{multi_loaded.parameters[i][0].name}")
            else:
                LOGGER.error(f"\u274c Attribute {attr} is not equal between {norma_loaded.name} and multinormalizer.{multi_loaded.parameters[i][0].name}")
                err = True

    if err:
        raise ValueError("Attributes are not equal between (at least) one of the individual normalizer and the multinormalizer")
    LOGGER.info("Loading from json seems to work\n")

    return normalizers_loaded, multi_loaded



def test_main(normalizer_list):

    data = create_dummy_dataset(nb_var=len(normalizer_list))
    normalizer_individual = []
    method_var_dict = {normalizer: ([f"var{i}"],{}) for i, normalizer in enumerate(normalizer_list)}
    if "BoxCoxScaler" in normalizer_list:
        method_var_dict["BoxCoxScaler"] = (method_var_dict["BoxCoxScaler"][0], {"lambda_": 0.5})
    if "YeoJohnsonScaler" in normalizer_list:
        method_var_dict["YeoJohnsonScaler"] = (method_var_dict["YeoJohnsonScaler"][0], {"lambda_": 0.3})

    LOGGER.info(f"Method var dict: {method_var_dict}")
    multinormalizer = st.MultiNormalizer(method_var_dict=method_var_dict)
    
    for normalizer in normalizer_list:
        normalizer_individual.append(st.create_normalizer_from_str(normalizer, inputs=method_var_dict[normalizer][1]))

    normalizers, multinormalizer = test_fit_data(normalizer_individual, multinormalizer, data)

    data_individual, data_multi = test_transform_data(normalizers, multinormalizer, data)

    data_individual, data_multi = test_inverse_transform_data(normalizers, multinormalizer, data_individual, data_multi, data)

    normalizers_dicts, multi_dict = test_save_as_dict(normalizers, multinormalizer)

    _, _ = test_load_from_dict(normalizers, multinormalizer, normalizers_dicts, multi_dict)

    filepaths = test_save_to_json(normalizers, multinormalizer, "./test_multi")

    _, _ = test_load_from_json(normalizers, multinormalizer, filepaths)


if __name__ == "__main__":
    setup_logger("tests_multi.log")

    normalizers = [n.name for n in st.Normalizer.__subclasses__() if not n.name == "MultiNormalizer"]
    LOGGER.info(f"Normalizers: {normalizers}")
    test_main(normalizer_list=normalizers)