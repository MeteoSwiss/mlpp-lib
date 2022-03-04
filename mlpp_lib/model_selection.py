import random
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
    ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[float] = None,
        gap: str = "10D",
        size_tolerance: float = 0.05,
        uniformity: Optional[str] = "month",
        uniformity_tolerance: float = 0.01,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = pd.Timedelta(gap)
        self.size_tolerance = size_tolerance
        self.uniformity = uniformity
        self.uniformity_tolerance = uniformity_tolerance

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
        n_splits = self.n_splits
        test_size = self.test_size if self.test_size is not None else 1 / n_splits
        test_interval = (X[-1] - X[0]) * test_size - self.gap
        which = combinations(range(n_splits), 1)
        which = np.array(list(which), dtype=int)[::-1]
        splits = np.zeros((len(which), n_splits), dtype=int)
        splits[np.arange(len(which))[None].T, which] = 1
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
                    gap_end = end + self.gap
                    split.loc[gap_start:gap_end] = 3
                start += test_interval + self.gap
            split.loc[X[-1] - self.gap : X[-1]] = 3
            if self._check_split(split, test_size):
                good_splits.append(split.values)
            if len(good_splits) == self.n_splits:
                break
        n_good_splits = len(good_splits)
        if n_good_splits < self.n_splits:
            raise RuntimeError(
                "we could not find enough valid splits... try change your input parameter?"
            )
        for good_split in good_splits:
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
            return False
        if wsd < self.uniformity_tolerance:
            return True
        else:
            return False
