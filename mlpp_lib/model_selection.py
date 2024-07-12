import json
import logging
import random
from copy import deepcopy
from itertools import combinations
from typing import Any, Mapping, Optional, Sequence, Type

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wasserstein_distance
from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


xindex = Type[Sequence or np.ndarray]


class DataSplitter:
    """A helper class for managing data partitioning in mlpp.

    Parameters
    ----------
    time_split: dict-like
        A mapping from partition name to the xarray indexer for the
        `forecast_reference_time` index dimension.

        The following notations are accepted:

        - mapping {partition name: (start, end)}, where start and end
            can be either datetime string representations (e.g. '2022-01-01')
            or datetime objects (e.g. `datetime(2022, 1, 1)`).
        - mapping {partition name: Sequence[labels]}, where labels can be
            either datetime string representations (e.g. '2022-01-01')
            or datetime objects (e.g. `datetime(2022, 1, 1)`).
        - mapping {partition name: partition fraction} where the fraction is
            a `float` type.

        And some combination of those are also accepted:

        -    {"train":              partition fraction},
              "val":                partition fraction},
              "test": (start, end) or Sequence[labels]}}

        If a mapping with partition fractions is provided, then
        one must also provide the `time_split_method` parameter.


    station_split: dict-like, optional
        A mapping from partition name to an xarray indexer for the
        `station` index dimension.

        The following notations are accepted:

        - mapping {partition name: Sequence[labels]}, where labels are station
            names (e.g. "LUG", "KLO", etc.)
        - mapping {partition name: partition fraction}, where the fraction is a
            `float` type.

        And one combination of those is also accepted:

        - {"train":                partition fraction},
            "val":                 partition fraction },
            "test":                Sequence[labels]}}

        If a mapping with partition fractions is provided, then
        one must also provide the `station_split_method` parameter.

    time_split_method: string, optional
        The method to split along time. Currently implemented are:
        - "sequential": splits the `time_index` sequentially based on
            the partition fractions

    station_split_method: string, optional
        The method to split along time. Currently implemented are:
        - "random": splits the `station_index` randomly based on
            the partition fractions


    Examples
    --------
    >>> splitter = DataSplitter(
    ...     time_split = {"train": 0.6, "val": 0.2, "test": 0.2},
    ...     time_split_method = "sequential"
    ... )
    ... train_features, train_targets = splitter.get_partition(
    ...     features, targets, partition="train"
    ... )
    """

    partitions: dict[str, dict[str, xindex]]
    time_split_methods = ["sequential"]
    station_split_methods = ["random", "sequential"]

    def __init__(
        self,
        time_split: dict[str, Any],
        station_split: Optional[dict[str, Any]],
        time_split_method: Optional[str] = None,
        station_split_method: Optional[str] = None,
        seed: Optional[int] = 10,
        time_dim_name: str = "forecast_reference_time",
    ):
        if not time_split.keys() == station_split.keys():
            raise ValueError(
                "Time split and station split must be defined "
                "with the same partitions!"
            )
        self.partition_names = list(time_split.keys())
        self._check_time(time_split, time_split_method)
        self._check_station(station_split, station_split_method)
        self.seed = seed
        self.time_dim_name = time_dim_name

    def fit(self, *args: xr.Dataset) -> Self:
        """Compute splits based on the input datasets.

        Parameters
        ----------
        *args: `xr.Dataset`
            The datasets defining the time and station indices.

        Returns
        -------
        self: `DataSplitter`
            The fitted DataSplitter instance.
        """

        self.time_index = args[0][self.time_dim_name].values.copy()
        self.station_index = args[0].station.values.copy()

        self._time_partitioning()
        self._station_partitioning()

        return self

    def get_partition(
        self, *args: xr.Dataset, partition=None, thinning: Optional[Mapping] = None
    ) -> tuple[xr.Dataset, ...]:
        """
        Select and return a partition based on the current values in
        the `partitions` attribute.

        Parameters
        ----------
        *args: `xr.Dataset`
            The dataset arguments from which we want to select a partition.
        partition: str
            The keyword for the partition, e.g. "train", "val" or "test".
        thinning: dict
            Mapping from dimension name to integer, indicating by how many times to reduce
            the number of labels for such dimension coordinate. May be used to reduce the size
            of an unnecessarily large dataset.

        Returns
        -------
        partition: tuple of `xr.Dataset`
            Subsets of the input datasets.
        """

        if partition is None:
            raise ValueError("Keyword argument `partition` must be provided.")

        if not hasattr(self, "time_index") or not hasattr(self, "station_index"):
            self = self.fit(*args)

        # avoid out-of-order indexing (leads to bad performance with xarray/dask)
        station_idx = self.partitions[partition]["station"]
        idx_loc = [pd.Index(self.station_index).get_loc(label) for label in station_idx]
        self.partitions[partition]["station"] = list(
            np.array(station_idx)[np.argsort(idx_loc)]
        )

        # apply thinning
        if thinning:
            indexers = {
                dim: coord[slice(None, None, thinning.get(dim, None))]
                for dim, coord in self.partitions[partition].items()
            }
        else:
            indexers = self.partitions[partition]

        res = tuple(ds.sel(indexers) for ds in args)
        for ds in args:
            ds.close()
        del args
        return res

    def _time_partitioning(self) -> None:

        # actual computation of splits

        if (
            self._time_defined
        ):  # provided time split is all valid xarray indexers (lists or slices)
            self._time_indexers = self.time_split
        else:
            self._time_indexers = {}
            if all([isinstance(v, float) for v in self.time_split.values()]):
                res = self._time_partition_method(self.time_split)
                self._time_indexers.update(res)
            else:  # mixed fractions and labels
                _time_split = self.time_split.copy()
                test_indexers = _time_split.pop("test")
                test_indexers = [t for t in test_indexers if t in self.time_index]
                self._time_indexers.update({"test": test_indexers})
                res = self._time_partition_method(_time_split)
                self._time_indexers.update(res)

        # assign indexers
        for partition in self.partition_names:
            idx = self._time_indexers[partition]
            idx = pd.to_datetime(idx)  # always convert to pandas datetime indices
            if len(idx) == 2:
                # convert slice to list of labels
                time_index = pd.to_datetime(self.time_index)
                idx = time_index[time_index.slice_indexer(start=idx[0], end=idx[1])]
            indexer = {self.time_dim_name: idx}
            if not hasattr(self, "partitions"):
                self.partitions = {p: {} for p in self.partition_names}
            self.partitions[partition].update(indexer)

    def _time_partition_method(self, fractions: Mapping[str, float]):
        time_index = [
            t for t in self.time_index if t not in self._time_indexers.get("test", [])
        ]
        if self.time_split_method == "sequential":
            return sequential_split(time_index, fractions)

    def _station_partitioning(self):
        """
        Compute station partitioning for this DataSplitter instance.
        """

        self._station_indexers: dict[str, xindex] = {}
        if not self._station_defined:
            if all([isinstance(v, float) for v in self.station_split.values()]):
                res = self._station_partition_method(self.station_split)
                self._station_indexers.update(res)
            else:  # mixed fractions and labels
                _station_split = self.station_split.copy()
                test_indexers = _station_split.pop("test")
                test_indexers = [s for s in test_indexers if s in self.station_index]
                self._station_indexers.update({"test": test_indexers})
                res = self._station_partition_method(_station_split)
                self._station_indexers.update(res)
        else:
            self._station_indexers.update(self.station_split)

        for partition in self.partition_names:
            indexer = {"station": self._station_indexers[partition]}
            if not hasattr(self, "partitions"):
                self.partitions = {p: {} for p in self.partition_names}
            self.partitions[partition].update(indexer)

    def _station_partition_method(
        self, fractions: Mapping[str, float]
    ) -> Mapping[str, np.ndarray]:

        station_index = [
            sta
            for sta in self.station_index
            if sta not in self._station_indexers.get("test", [])
        ]
        if self.station_split_method == "random":
            out = random_split(station_index, fractions, seed=self.seed)
        elif self.station_split_method == "sequential":
            out = sequential_split(station_index, fractions)
        return out

    def _check_time(self, time_split: dict, time_split_method: str):

        if any([isinstance(v, float) for v in time_split.values()]):
            if time_split_method is None:
                raise ValueError(
                    "`time_split_method` must be provided if the time "
                    "splits are provided as fractions!"
                )
            self._time_defined = False
            if time_split_method not in self.time_split_methods:
                raise ValueError(
                    f"Invalid time split method: {time_split_method}. "
                    f"Must be one of {self.time_split_methods}."
                )
        else:
            self._time_defined = True

        self.time_split = time_split
        self.time_split_method = time_split_method

    def _check_station(self, station_split: dict, station_split_method: str):

        if any([isinstance(v, float) for v in station_split.values()]):
            if station_split_method is None:
                raise ValueError(
                    "`station_split_method` must be provided if the "
                    "station splits are provided as fractions!"
                )
            self._station_defined = False
            if station_split_method not in self.station_split_methods:
                raise ValueError(
                    f"Invalid station split method: {station_split_method}. "
                    f"Must be one of {self.station_split_methods}."
                )
        else:
            self._station_defined = True

        self.station_split = station_split
        self.station_split_method = station_split_method

    @classmethod
    def from_dict(cls, splits: dict) -> Self:
        time_split = {k: v["forecast_reference_time"] for k, v in splits.items()}
        station_split = {k: v["station"] for k, v in splits.items()}
        splitter = cls(time_split, station_split)
        splitter._time_defined = True
        splitter._station_defined = True
        splitter._time_partitioning()
        splitter._station_partitioning()
        return splitter

    def to_dict(self, sort_values=False):
        if not hasattr(self, "time_index") or not hasattr(self, "station_index"):
            raise ValueError(
                "DataSplitter wasn't applied on any data yet, run `fit` first."
            )
        if not hasattr(self, "partitions"):
            self._time_partitioning()
            self._station_partitioning()
        partitions = deepcopy(self.partitions)
        for split_key, split_dict in partitions.items():
            for dim, value in split_dict.items():
                if isinstance(value, slice):
                    partitions[split_key][dim] = [str(value.start), str(value.stop)]
                elif hasattr(value, "tolist"):
                    partitions[split_key][dim] = value.astype(str).tolist()
                    if sort_values:
                        partitions[split_key][dim] = sorted(partitions[split_key][dim])
        return partitions

    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)

    def save_json(self, out_fn: str) -> None:
        out_dict = self.to_dict()
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)


def sequential_split(
    index: xindex,
    split_fractions: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Split an input index array sequentially"""
    assert np.isclose(sum(split_fractions.values()), 1.0)

    n_samples = len(index)
    partitions = list(split_fractions.keys())
    fractions = np.array(list(split_fractions.values()))

    indices = np.floor(np.cumsum(fractions)[:-1] * n_samples).astype(int)
    sub_arrays = np.split(index, indices)
    return dict(zip(partitions, sub_arrays))


def random_split(
    index: xindex,
    split_fractions: Mapping[str, float],
    seed: int = 10,
) -> dict[str, np.ndarray]:
    """Split an input index array randomly"""
    rng = np.random.default_rng(np.random.PCG64(seed))

    assert np.isclose(sum(split_fractions.values()), 1.0)

    n_samples = len(index)
    partitions = list(split_fractions.keys())
    fractions = np.array(list(split_fractions.values()))

    shuffled_index = rng.permutation(index)
    indices = np.floor(np.cumsum(fractions)[:-1] * n_samples).astype(int)
    sub_arrays = np.split(shuffled_index, indices)
    return dict(zip(partitions, sub_arrays))


def _split_list(
    samples: list,
    split_ratios: list,
    shuffle_samples: bool = False,
    seed: Optional[int] = None,
) -> list:
    """Split list into subsets according to given split ratios."""
    split_ratios = np.array(split_ratios)
    split_ratios /= split_ratios.sum()
    if seed is not None:
        random.seed(seed)
    if shuffle_samples:
        random.shuffle(samples)

    n_samples = len(samples)
    split_data = []
    cum_fraction = 0
    for i, _ in enumerate(split_ratios):
        start = int(cum_fraction * n_samples)
        cum_fraction += split_ratios[i]
        end = int(cum_fraction * n_samples)
        split_data.append(sorted(samples[start:end]))

    return split_data


def train_test_split(
    samples: list, sample_labels: list, test_size: float, **split_kwargs
) -> list:
    """Split list into train and test sets."""
    split_ratios = [1 - test_size, test_size]

    if sample_labels is None:
        return _split_list(samples, split_ratios, **split_kwargs)

    if not len(sample_labels) == len(samples):
        raise ValueError("'samples' and 'sample_labels' must have the same length")

    unique_labels = list(set(sample_labels))
    n_labels = len(unique_labels)
    label_data = [[] for _ in range(n_labels)]
    for label, value in zip(sample_labels, samples):
        label_data[unique_labels.index(label)].append(value)

    split_label_data = [
        _split_list(sublist, split_ratios, **split_kwargs) for sublist in label_data
    ]

    data_split = []
    for i, _ in enumerate(split_ratios):
        data_split.append([item for sublist in split_label_data for item in sublist[i]])

    return data_split


class UniformTimeSeriesSplit:
    """Time Series cross-validator ...

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    test_size : (Optional) float
        The size (as fraction) of the test set.
    interval : (Optional), str
        The length of a cross-validation interval of the as a duration string (e.g. 180D).
        When not set, the interval is effectively equals to the length of the test set.
        Example: with five years of data and test_size=0.2, if interval is not set we will
        have maximum of five splits (that are also exclusive and are therefore "folds"). But if
        the interval is six months, we can have 5 x 5 = 25 splits in total. In this case, however,
        they will be non-exclusive.
    gap : str, default="10D"
        Length of a gap between sets as a duration string. Can be used to
        preserve independency between sets.
    size_tolerance : float, default=0.05
        Because of gaps (and interval) lengths, the size of the test
        set may not be exactly the one specified in `test_size`. This argument
        sets the tolerance.
    uniformity: str, default="month"
        Label used to check for uniformity.
    uniformity_tolerance: float, default=0.01
        Maximum tolerated distance from a uniform distribution,
        calculated using Wasserstein distance.

    Note
    ----
    The use of this feature often requires some trial and error.


    ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[float] = None,
        interval: Optional[str] = None,
        gap: str = "10D",
        size_tolerance: float = 0.05,
        uniformity: Optional[str] = "month",
        uniformity_tolerance: float = 0.01,
        random_state: int = 1234,
    ):

        if test_size:
            if (n_splits > 1 / test_size) and interval is None:
                raise ValueError(
                    f"Cannot return {n_splits} splits. A maximum of  1 / {test_size}"
                    f" splits can be returned if test_size={test_size} and"
                    " the interval argument is not set."
                )

        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = pd.Timedelta(gap)
        self.interval = pd.Timedelta(interval) if interval else None
        self.size_tolerance = size_tolerance
        self.uniformity = uniformity
        self.uniformity_tolerance = uniformity_tolerance
        self.random_state = random_state

    def split(
        self,
        X,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples)
            Time stamps of training data.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        X = pd.to_datetime(X)

        self.failed_uniformity = 0
        self.failed_size = 0
        n_splits = self.n_splits
        test_size = self.test_size if self.test_size is not None else 1 / n_splits

        # if interval is explicitely defined by the user
        if self.interval:
            test_interval = self.interval
            n_intervals = int((X[-1] - X[0]) / test_interval)
            n_test_intervals = max(int(n_intervals * test_size), 1)

        # if interval is simply the length of the test set
        else:
            test_interval = (X[-1] - X[0]) * test_size - self.gap
            n_intervals = n_splits
            n_test_intervals = 1

        which = list(combinations(range(n_intervals), n_test_intervals))
        which = np.array(which, dtype=int)[::-1]
        splits = np.zeros((len(which), n_intervals), dtype=int)
        splits[np.arange(len(which))[None].T, which] = 1

        if len(splits) < n_splits:
            raise ValueError(
                f"The number of possible combinations is {len(splits)}"
                f" but the requested number of splits is {n_splits}."
            )

        rng = np.random.default_rng(self.random_state)
        rng.shuffle(splits)

        good_splits = []
        for choice_array in splits:
            split = pd.Series(-1, index=X, dtype=int)
            start = X[0]
            for idx, choice in enumerate(choice_array):
                next_choice = choice_array[min(idx + 1, len(choice_array) - 1)]
                end = start + test_interval
                split.loc[start:end] = choice
                if next_choice != choice:
                    gap_start = end
                    gap_end = gap_start + self.gap
                    split.loc[gap_start:gap_end] = 3
                    start += self.gap
                start += test_interval
            split.loc[X[-1] - self.gap : X[-1]] = 3
            if self._check_split(split, test_size):
                good_splits.append(split.values)
            if len(good_splits) == self.n_splits:
                break
        n_good_splits = len(good_splits)
        if n_good_splits < self.n_splits:
            raise RuntimeError(
                "We could not find enough valid splits! "
                f"{len(splits)} splits were checked but only {n_good_splits} passed... "
                f"{self.failed_size} failed size checks, "
                f"{self.failed_uniformity} uniformity checks. "
                "Try change your input parameter?"
            )
        for good_split in good_splits[:n_splits]:
            train_idx = np.where(good_split == 0)[0].tolist()
            test_idx = np.where(good_split == 1)[0].tolist()
            yield train_idx, test_idx

    def _check_split(self, split, test_size):
        """Check that desired criteria are met for a given set"""
        proportion = self._check_proportions(split, test_size)
        uniformity = self._check_uniformity(split, test_size)
        return proportion and uniformity

    def _check_proportions(self, split, test_size):
        """Check that proportions are respected between sets"""
        train_times = split[split == 0]
        train_proportion = len(train_times) / len(split)
        if np.isclose(train_proportion, 1 - test_size, atol=self.size_tolerance):
            return True
        else:
            self.failed_size += 1
            return False

    def _check_uniformity(self, split, test_size):
        """Check that timestamps are uniformly distributed in the smaller set."""
        if self.uniformity is None:
            return True
        id_small_set = np.argmin((1 - test_size, test_size))
        smaller_set = split[split == id_small_set]
        index = getattr(smaller_set.index, self.uniformity)
        counts = smaller_set.groupby(index).count()
        freq = counts / smaller_set.count()
        uniform_dist = np.ones(12) / 12
        try:
            wsd = wasserstein_distance(freq, uniform_dist)
        except ValueError:
            self.failed_uniformity += 1
            return False
        if wsd < self.uniformity_tolerance:
            return True
        else:
            self.failed_uniformity += 1
            return False
