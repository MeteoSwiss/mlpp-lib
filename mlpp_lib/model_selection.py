from multiprocessing.sharedctypes import Value
import random
from re import A
from typing import Optional
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


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
