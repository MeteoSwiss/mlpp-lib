from typing import Mapping, Optional, Any
from dataclasses import dataclass
import json
import yaml

import pandas as pd
import numpy as np
import pytest

import mlpp_lib.model_selection as ms


def check_splits(splits: dict):
    """
    Check if the data splits are valid.

    Args:
        splits (dict): A dictionary containing train, val, and test splits.
                       Each split should contain 'station' and 'forecast_reference_time' keys.

    Raises:
        AssertionError: If there are overlapping stations or forecast reference times between splits.
    """
    train_stations = set(splits["train"]["station"])
    val_stations = set(splits["val"]["station"])
    test_stations = set(splits["test"]["station"])

    train_reftimes = set(splits["train"]["forecast_reference_time"])
    val_reftimes = set(splits["val"]["forecast_reference_time"])
    test_reftimes = set(splits["test"]["forecast_reference_time"])

    assert len(train_stations & test_stations) == 0, "Train and test stations overlap."
    assert len(train_stations & val_stations) == 0, "Train and val stations overlap."
    assert len(val_stations & test_stations) == 0, "Val and test stations overlap."
    assert (
        len(train_reftimes & test_reftimes) == 0
    ), "Train and test forecast reference times overlap."
    assert (
        len(train_reftimes & val_reftimes) == 0
    ), "Train and val forecast reference times overlap."
    assert (
        len(val_reftimes & test_reftimes) == 0
    ), "Val and test forecast reference times overlap."


@dataclass
class ValidDataSplitterOptions:

    time: str
    station: str

    reftimes = pd.date_range("2018-01-01", "2018-03-31", freq="24H")
    stations = [chr(i) * 3 for i in range(ord("A"), ord("Z"))]

    def __post_init__(self):

        if self.time == "fractions":
            self.time_split = {"train": 0.6, "val": 0.2, "test": 0.2}
            self.time_split_method = "sequential"
        elif self.time == "lists":
            self.time_split = self.time_split_lists()
            self.time_split_method = None
        elif self.time == "slices":
            self.time_split = self.time_split_slices()
            self.time_split_method = None
        elif self.time == "mixed":
            self.time_split = {"train": 0.7, "val": 0.3, "test": self.reftimes[-10:]}
            self.time_split_method = "sequential"

        if self.station == "fractions":
            self.station_split = {"train": 0.6, "val": 0.2, "test": 0.2}
            self.station_split_method = "random"
        elif self.station == "lists":
            self.station_split = self.station_split_lists()
            self.station_split_method = None
        elif self.station == "mixed":
            self.station_split = {"train": 0.7, "val": 0.3, "test": self.stations[-5:]}
            self.station_split_method = "random"
        elif self.station == "cloud_mixed":
            with open("tests/stations_cloud.json", "r") as f:
                self.stations = json.load(f)
            with open("tests/stations_cloud_splits.yaml", "r") as f:
                self.station_split = yaml.safe_load(f)
            self.station_split_method = "random"

    def time_split_lists(self):
        frac = {"train": 0.6, "val": 0.2, "test": 0.2}
        return ms.sequential_split(self.reftimes, frac)

    def time_split_slices(self):
        out = {
            "train": [self.reftimes[0], self.reftimes[30]],
            "val": [self.reftimes[31], self.reftimes[40]],
            "test": [self.reftimes[41], self.reftimes[50]],
        }
        return out

    def station_split_lists(self):
        frac = {"train": 0.6, "val": 0.2, "test": 0.2}
        return ms.random_split(self.stations, frac)

    def pytest_id(self):
        return f"time: {self.time}, station: {self.station}"


class TestDataSplitter:

    scenarios = [
        ValidDataSplitterOptions(time="fractions", station="lists"),
        ValidDataSplitterOptions(time="slices", station="fractions"),
        ValidDataSplitterOptions(time="lists", station="fractions"),
        ValidDataSplitterOptions(time="lists", station="mixed"),
        ValidDataSplitterOptions(time="mixed", station="fractions"),
        ValidDataSplitterOptions(time="mixed", station="mixed"),
        ValidDataSplitterOptions(time="mixed", station="cloud_mixed"),
    ]

    @pytest.mark.parametrize(
        "options", scenarios, ids=ValidDataSplitterOptions.pytest_id
    )
    def test_valid_split(self, options, features_dataset, targets_dataset):
        splitter = ms.DataSplitter(
            options.time_split,
            options.station_split,
            options.time_split_method,
            options.station_split_method,
        )
        splits = splitter.fit(features_dataset).to_dict()

        check_splits(splits)

    @pytest.mark.parametrize(
        "options", scenarios, ids=ValidDataSplitterOptions.pytest_id
    )
    def test_get_partition(self, options, features_dataset, targets_dataset):
        splitter = ms.DataSplitter(
            options.time_split,
            options.station_split,
            options.time_split_method,
            options.station_split_method,
        )
        train_features, train_targets = splitter.get_partition(
            features_dataset, targets_dataset, partition="train"
        )
        val_features, val_targets = splitter.get_partition(
            features_dataset, targets_dataset, partition="val"
        )
        test_features, test_targets = splitter.get_partition(
            features_dataset, targets_dataset, partition="test"
        )

    @pytest.mark.parametrize(
        "options", scenarios, ids=ValidDataSplitterOptions.pytest_id
    )
    def test_serialization(self, options, features_dataset, targets_dataset, tmp_path):
        fn = f"{tmp_path}/splitter.json"
        splitter = ms.DataSplitter(
            options.time_split,
            options.station_split,
            options.time_split_method,
            options.station_split_method,
        )
        splitter.get_partition(features_dataset, targets_dataset, partition="train")
        splitter.save_json(fn)
        new_splitter = ms.DataSplitter.from_json(fn)
        for split_key, split_dict in splitter.partitions.items():
            for dim, value in split_dict.items():
                new_value = new_splitter.partitions[split_key][dim]
                np.testing.assert_array_equal(value, new_value)

    # test invalid arguments
    def test_time_split_method_required(self):
        time_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        station_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        with pytest.raises(ValueError) as excinfo:
            splitter = ms.DataSplitter(
                time_split, station_split, station_split_method="random"
            )
        assert (
            str(excinfo.value)
            == "`time_split_method` must be provided if the time splits are provided as fractions!"
        )

    def test_station_split_method_required(self):
        time_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        station_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        with pytest.raises(ValueError) as excinfo:
            splitter = ms.DataSplitter(
                time_split, station_split, time_split_method="sequential"
            )
        assert (
            str(excinfo.value)
            == "`station_split_method` must be provided if the station splits are provided as fractions!"
        )

    def test_station_split_keys_invalid(self):
        time_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        station_split = {"train": 0.6, "val": 0.2}
        with pytest.raises(ValueError) as excinfo:
            splitter = ms.DataSplitter(
                time_split,
                station_split,
                time_split_method="sequential",
                station_split_method="random",
            )
        assert (
            str(excinfo.value)
            == "Time split and station split must be defined with the same partitions!"
        )

    def test_time_split_method_invalid(self):
        time_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        station_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        time_split_method = "invalid_method"
        station_split_method = "random"
        with pytest.raises(ValueError) as excinfo:
            splitter = ms.DataSplitter(
                time_split,
                station_split,
                time_split_method=time_split_method,
                station_split_method=station_split_method,
            )
        assert (
            str(excinfo.value)
            == f"Invalid time split method: {time_split_method}. Must be one of {ms.DataSplitter.time_split_methods}."
        )

    def test_station_split_method_invalid(self):
        time_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        station_split = {"train": 0.6, "val": 0.2, "test": 0.2}
        time_split_method = "sequential"
        station_split_method = "invalid_method"
        with pytest.raises(ValueError) as excinfo:
            splitter = ms.DataSplitter(
                time_split,
                station_split,
                time_split_method=time_split_method,
                station_split_method=station_split_method,
            )
        assert (
            str(excinfo.value)
            == f"Invalid station split method: {station_split_method}. Must be one of {ms.DataSplitter.station_split_methods}."
        )


def test_sequential_split(features_dataset):
    index = features_dataset.forecast_reference_time.values[:50]
    split_fractions = {"train": 0.6, "valid": 0.2, "test": 0.2}
    result = ms.sequential_split(index, split_fractions)
    assert len(result) == 3
    assert len(result["train"]) == 30
    assert len(result["valid"]) == 10
    assert len(result["test"]) == 10
    split_fractions = {"train": 0.6, "valid": 0.2, "test": 0.3}
    with pytest.raises(AssertionError):
        ms.sequential_split(index, split_fractions)


def test_random_split(features_dataset):
    index = features_dataset.station.values[:30]
    split_fractions = {"train": 0.5, "valid": 0.3, "test": 0.2}
    result = ms.sequential_split(index, split_fractions)
    assert len(result) == 3
    assert len(result["train"]) == 12
    assert len(result["valid"]) == 8
    assert len(result["test"]) == 5
    split_fractions = {"train": 0.6, "valid": 0.2, "test": 0.3}
    with pytest.raises(AssertionError):
        ms.sequential_split(index, split_fractions)


def get_split_lengths(test_size, n):
    split_lengths = []
    split_ratios = [1 - test_size, test_size]
    start = 0
    for ratio in split_ratios:
        end = start + ratio
        split_lengths.append(int(end * n) - int(start * n))
        start = end
    return split_lengths


def test_train_test_split():
    """"""
    n_samples = 101
    test_size = 0.2
    n_labels = 7

    samples = list(range(n_samples))
    labels = np.random.randint(0, n_labels, n_samples)

    sample_split = ms.train_test_split(samples, labels, test_size)
    assert len(sample_split) == 2
    assert len([item for sublist in sample_split for item in sublist]) == n_samples

    for label in set(labels):
        subdata = [s for l, s in zip(labels, samples) if l == label]
        split_lengths = get_split_lengths(test_size, len(subdata))
        assert sum([s in sample_split[0] for s in subdata]) == split_lengths[0]
        assert sum([s in sample_split[1] for s in subdata]) == split_lengths[1]


def test_time_series_cv():
    """"""
    n_splits = 5
    reftimes = np.arange("2016-01-01", "2021-01-01", dtype="datetime64[12h]").astype(
        "datetime64[ns]"
    )
    cv = ms.UniformTimeSeriesSplit(n_splits)
    for n, (train, test) in enumerate(cv.split(reftimes)):
        assert isinstance(train, list)
        assert isinstance(test, list)
        assert np.isclose(len(train) / len(reftimes), 1 - 1 / n_splits, atol=0.05)
        assert np.isclose(len(test) / len(reftimes), 1 / n_splits, atol=0.05)
    assert n_splits == (n + 1)
