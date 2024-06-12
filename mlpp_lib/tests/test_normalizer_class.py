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
            f"Values are not equal: {ds1} != {ds2}"
        return False

    ds1_converted = ds1#.astype('float64')
    ds2_converted = ds2#.astype('float64')
    
    try:
        # Use xarray's testing function to check for equality with numerical tolerance
        xr.testing.assert_allclose(ds1_converted, ds2_converted)
        return True
    except AssertionError as e:
        # If there are differences, catch the exception and print the differences
        print("Datasets are not equal. Differences:")
        print(e)
        # Optionally, print a detailed difference
        print("\nDetailed differences:")
        ds1_vars = set(ds1.data_vars)
        ds2_vars = set(ds2.data_vars)
        
        # Check for differences in variables
        for var in ds1_vars.union(ds2_vars):
            if var not in ds1_vars:
                print(f"Variable {var} is missing in the first dataset.")
            elif var not in ds2_vars:
                print(f"Variable {var} is missing in the second dataset.")
            else:
                array1 = ds1_converted[var].values
                array2 = ds2_converted[var].values
                if not np.allclose(array1, array2):
                    diff = array1 - array2
                    print(f"Differences in variable '{var}':")
                    print(diff)
        return False


def create_dummy_dataset(nb_var=2):
    data = xr.Dataset(
        {
            f"var{i}": (("x", "y"), np.arange(4**i, 4**(i+1)).reshape(2, 2))\
            for i in range(nb_var)
        },
        coords={"x": [10, 20], "y": [1, 2]},
    )
    return data


def test_fit_transform(n1, n2, nb_var=2):
    data = create_dummy_dataset(nb_var=nb_var)
    LOGGER.info(f"Data: {data}")

    n1.fit(data)
    ds1 = n1.transform(data)[0]

    n2.fit(data)
    ds2 = n2.transform(data)[0]

    # check if the 2 datasets are equal

    if not check_equality(ds1, ds2):
        raise ValueError("The 2 datasets are not equal")
    else:
        LOGGER.info("The 2 datasets are equal")
    
    LOGGER.info("Fitting and transforming test passed")

    return n1, n2, data, ds1, ds2


def test_todict(n1, n2):

    out_dict1 = n1.to_dict()
    out_dict2 = n2.to_dict()
    out_dict2_tmp = out_dict2[list(out_dict2.keys())[0]]

    for key in out_dict1.keys():
        if out_dict1[key] != out_dict2_tmp[key]:
            LOGGER.info(f"out_dict1: {out_dict1[key]}")
            LOGGER.info(f"out_dict2: {out_dict2_tmp[key]}")
            raise ValueError("The 2 dictionaries are not equal for key: {key}")
    LOGGER.info("The 2 dictionaries are equal")
    LOGGER.info("Saving to dictionary test passed")

    return out_dict1, out_dict2


def test_fromdict(out_dict1, out_dict2, n1, n2):

    # use the method directly without creating a class instance 
    n3 = type(n1).from_dict(out_dict1)
    n4 = type(n2).from_dict(out_dict2)

    err = False
    for attr in get_class_attributes(n1):
        if check_equality(getattr(n1, attr), getattr(n3, attr)):
            LOGGER.info(f"n1 {attr} has stayed the same")
        else:
            LOGGER.info(f"n1 {attr} has changed")
            err = True

    if err:
        LOGGER.info("n1 and n3 are not equal. Check the logs for more details. Continuing with the test...")
    else:
        LOGGER.info("Loading from dictionary test passed for n1")

    # Check for n2 now
    assert n4.method_vars_list == n2.method_vars_list, f"n2 and n4 method_vars_list are different: {n2.method_vars_list} != {n4.method_vars_list}"
    err = False
    
    for i, _ in enumerate(n2.method_vars_list):
        mth_n2 = n2.method_vars_list[i][0]
        mth_n4 = n4.method_vars_list[i][0]
        for attr in get_class_attributes(mth_n2):
            if check_equality(getattr(mth_n2, attr), getattr(mth_n4, attr)):
                LOGGER.info(f"n2 {attr} has stayed the same")
            else:
                LOGGER.info(f"n2 {attr} has changed")
                err = True

    if err:
        raise ValueError("n2 and n4 are not equal. Check the logs for more details.")

    LOGGER.info("Loading from dictionary test passed for n2")



def test_load_json(json1, json2, n1, n2):

    n5 = type(n1).from_json(json1)
    n6 = type(n2).from_json(json2)

    err = False

    for attr in get_class_attributes(n1):
        if check_equality(getattr(n1, attr), getattr(n5, attr)):
            LOGGER.info(f"n1 {attr} has stayed the same")
        else:
            LOGGER.info(f"n1 {attr} has changed")
            err = True

    # Check for n2 now

    assert n6.method_vars_list == n2.method_vars_list, f"n2 and n6 method_vars_list are different: {n2.method_vars_list} != {n6.method_vars_list}"
    err = False

    for i, _ in enumerate(n2.method_vars_list):
        mth_n2 = n2.method_vars_list[i][0]
        mth_n6 = n6.method_vars_list[i][0]
        for attr in get_class_attributes(mth_n2):
            if check_equality(getattr(mth_n2, attr), getattr(mth_n6, attr)):
                LOGGER.info(f"n2 {attr} has stayed the same")
            else:
                LOGGER.info(f"n2 {attr} has changed")
                err = True

    if err:
        raise ValueError("n2 and n6 are not equal. Check the logs for more details.")
    
    LOGGER.info("Loading from json test passed for n2")


def test_inv_transform(n1, n2, data, ds1, ds2):

    ds1 = n1.inverse_transform(ds1)[0]
    ds2 = n2.inverse_transform(ds2)[0]

    err = False

    if not check_equality(ds1, data):
        LOGGER.info("Data and ds1 are not equal")
        err = True
    else:
        LOGGER.info("Data and ds1 are equal")

    if not check_equality(ds2, data):
        LOGGER.info("Data and ds2 are not equal")
        err = True
    else:
        LOGGER.info("Data and ds2 are equal")

    if err:
        raise ValueError("The inverse transform test failed")
    else:
        LOGGER.info("The inverse transform test passed")




def test_standardizer():

    LOGGER.info("Testing Standardizer")

    # create a standardizer in 2 different ways
    n1 = st.Standardizer()
    n2 = st.MultiNormalizer(method_var_dict={'Standardizer': ["a", "b"]})

    # Test the fit and transform functions
    
    LOGGER.info("Testing fit and transform functions")
    n1, n2, data, ds1, ds2 = test_fit_transform(n1, n2)

    # Test the saving

    LOGGER.info("Testing to_dict function")
    out_dict1, out_dict2 = test_todict(n1, n2)

    # test loading from a dictionary

    LOGGER.info("Testing from_dict function")
    test_fromdict(out_dict1, out_dict2, n1, n2)

    # test saving to json

    LOGGER.info("Testing save_json function")
    n1.save_json("./test_norma.json")
    n2.save_json("./test_norma2.json")

    LOGGER.info("Saving to json test passed")

    # test loading from json

    LOGGER.info("Testing load_json function")
    test_load_json("./test_norma.json", "./test_norma2.json", n1, n2)

    # test_inverse_transform

    LOGGER.info("Testing inverse_transform function")
    test_inv_transform(n1, n2, data, ds1, ds2)


def test_minmaxscaler():

    LOGGER.info("\n")
    LOGGER.info("Testing MinMaxScaler")

    n1 = st.MinMaxScaler()
    n2 = st.MultiNormalizer(method_var_dict={'MinMaxScaler': ["a", "b"]})

    # Test the fit and transform functions
    
    LOGGER.info("Testing fit and transform functions")
    n1, n2, data, ds1, ds2 = test_fit_transform(n1, n2)

    # Test the saving

    LOGGER.info("Testing to_dict function")
    out_dict1, out_dict2 = test_todict(n1, n2)

    # test loading from a dictionary

    LOGGER.info("Testing from_dict function")
    test_fromdict(out_dict1, out_dict2, n1, n2)

    # test saving to json

    LOGGER.info("Testing save_json function")
    n1.save_json("./test_minmax.json")
    n2.save_json("./test_minmax2.json")

    LOGGER.info("Saving to json test passed")

    # test loading from json

    LOGGER.info("Testing load_json function")
    test_load_json("./test_minmax.json", "./test_minmax2.json", n1, n2)

    # test_inverse_transform

    LOGGER.info("Testing inverse_transform function")
    test_inv_transform(n1, n2, data, ds1, ds2)


def test_multiple_normalizers(normalizers_list):

    LOGGER.info("\n")
    LOGGER.info("Testing Multiple Normalizers")

    n1s = [st.create_normalizer_from_str(normalizer) for normalizer in normalizers_list]
    n2 = st.MultiNormalizer(method_var_dict={normalizer: [f"var{i}"] for i, normalizer in enumerate(normalizers_list)})

    # Test the fit and transform functions
    
    LOGGER.info("Testing fit and transform functions")
    n1, n2, data, ds1, ds2 = test_fit_transform(n1, n2, nb_var=4)

    # Test the saving

    LOGGER.info("Testing to_dict function")
    out_dict1, out_dict2 = test_todict(n1, n2)

    # test loading from a dictionary

    LOGGER.info("Testing from_dict function")
    test_fromdict(out_dict1, out_dict2, n1, n2)

    # test saving to json

    LOGGER.info("Testing save_json function")
    n1.save_json("./test_multiple.json")
    n2.save_json("./test_multiple2.json")

    LOGGER.info("Saving to json test passed")

    # test loading from json

    LOGGER.info("Testing load_json function")
    test_load_json("./test_multiple.json", "./test_multiple2.json", n1, n2)

    # test_inverse_transform

    LOGGER.info("Testing inverse_transform function")
    test_inv_transform(n1, n2, data, ds1, ds2)







def main():
    setup_logger("./test.log")
    test_standardizer()
    test_minmaxscaler()


if __name__ == "__main__":
    main()
    