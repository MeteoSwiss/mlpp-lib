from mlpp_lib.model_selection import DataSplitter
import logging
import xarray as xr
import numpy as np
import json
import pandas as pd
import yaml
import copy

LOGGER = logging.getLogger(__name__)

def setup_logger(log_file, level="INFO", format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"):
    logging.basicConfig(
        level=logging.getLevelName(level),
        format=format,
        datefmt=datefmt,
        filename=log_file,
        filemode="w",
    )

def ceck_split_args(split_args, feature_ds):
    """Check the split arguments and adjust to the format expected by DataSplitter."""
    time_dim = split_args.get("time_dim_name", "forecast_reference_time")
    for split in ["train", "val", "test"]:
        
        if split not in split_args["time_split"]:
            raise ValueError(f"Split {split} is missing in time_split.")
        if split_args["time_split"][split] is None:
            split_args["time_split"][split] = pd.to_datetime(feature_ds[time_dim])

        if isinstance(split_args["time_split"][split], dict):
            # convert to slice
            split_args["time_split"][split] = (
                split_args["time_split"][split].get("start"),
                split_args["time_split"][split].get("end"),
            )

        if isinstance(split_args["time_split"][split], list):
            # expand the list of timestamps to all reftimes in the months

            times_filtered = []
            for year_month in split_args["time_split"][split]:
                filter_year, filter_month = year_month.split("-")
                times = pd.to_datetime(feature_ds[time_dim])
                times = times[times.year == int(filter_year)]
                times = times[times.month == int(filter_month)]
                times_filtered += times.tolist()
            split_args["time_split"][split] = times_filtered
    return split_args


def compare_dictionaries(dict1, dict2, parent_key=''):
    """
    Compare two multi-level dictionaries and log the differences if they are not equal.
    
    Args:
        dict1 (dict): The first dictionary to compare.
        dict2 (dict): The second dictionary to compare.
        parent_key (str): The parent key path used for logging nested keys.
        
    Returns:
        bool: True if dictionaries are equal, False otherwise.
    """
    are_equal = True
    
    # Check keys in dict1
    for key in dict1:
        full_key = f"{parent_key}.{key}" if parent_key else key
        if key not in dict2:
            LOGGER.error(f"Key '{full_key}' found in dict1 but not in dict2.")
            are_equal = False
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # Recursively compare nested dictionaries
                if not compare_dictionaries(dict1[key], dict2[key], full_key):
                    are_equal = False
            elif dict1[key] != dict2[key]:
                LOGGER.error(f"Difference found at key '{full_key}': dict1[{full_key}] = {dict1[key]}, dict2[{full_key}] = {dict2[key]}")
                are_equal = False

    # Check keys in dict2 not in dict1
    for key in dict2:
        full_key = f"{parent_key}.{key}" if parent_key else key
        if key not in dict1:
            LOGGER.error(f"Key '{full_key}' found in dict2 but not in dict1.")
            are_equal = False
            
    return are_equal


def read_json(in_fn: str) -> dict:
    LOGGER.info(f"read: {in_fn}")
    with open(in_fn, "r") as f:
        in_data = json.load(f)
    return in_data



def test_datasplitter(seed=42, station_split: dict[str, float] = None, time_split: dict[str, float] = None):

    dataset1 = xr.open_zarr("/scratch/lpoulain/data/wind/features.zarr")
    dataset2 = xr.open_zarr("/scratch/ned/mlpp/data/wind/features.zarr")
    assert all(dataset1["forecast_reference_time"].values == dataset2["forecast_reference_time"].values)
    filtered_stations_json = "/users/lpoulain/mlpp-workflows/artifacts/wind/hourly_wind_speed/qa/filtered_stations.json"

    filtered_stations = read_json(filtered_stations_json)
    stations1 = [x for x in dataset1.station.values if x not in filtered_stations]
    stations2 = [x for x in dataset2.station.values if x not in filtered_stations]
    features_ds1 = dataset1.sel(station=stations1)
    features_ds2 = dataset2.sel(station=stations2)

    time_split_copy = copy.deepcopy(time_split)
    time_split1 = ceck_split_args(time_split, features_ds1)
    time_split2 = ceck_split_args(time_split_copy, features_ds2)

    splitter1 = DataSplitter(
            time_split=time_split1["time_split"],
            station_split=station_split,
            time_split_method="sequential",
            station_split_method="random",
            seed=seed).fit(features_ds1)
    
    splitter2 = DataSplitter(
            time_split=time_split2["time_split"],
            station_split=station_split,
            time_split_method="sequential",
            station_split_method="random",
            seed=seed).fit(features_ds2)
    
    # Check that the two splitters are equal

    partitions1 = splitter1.to_dict()
    partitions2 = splitter2.to_dict()

    equal = compare_dictionaries(partitions1, partitions2)
    if equal:
        LOGGER.info("DataSplitter objects are equal.")
    else:
        LOGGER.error("DataSplitter objects are not equal.")

    partition1_sorted = splitter1.to_dict(sort_values=True)
    partition2_sorted = splitter2.to_dict(sort_values=True)

    equal = compare_dictionaries(partition1_sorted, partition2_sorted)
    if equal:
        LOGGER.info("DataSplitter objects are equal after sorting.")
    else:
        LOGGER.error("DataSplitter objects are not equal after sorting.")


if __name__ == "__main__":
    setup_logger("test_datasplitter.log")
    station_split = {"train": 0.8, "val": 0.1, "test": 0.1}
    with open("/users/lpoulain/mlpp-lib/mlpp_lib/tests/test_time_split.yaml", "r") as f:
        time_split = yaml.load(f, Loader=yaml.FullLoader)

    test_datasplitter(station_split=station_split, time_split=time_split)
    LOGGER.info("DataSplitter test completed successfully.")
    

    
