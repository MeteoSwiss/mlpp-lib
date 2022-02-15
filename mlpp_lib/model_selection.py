from multiprocessing.sharedctypes import Value
import random
from typing import Optional, Any
import time
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def split_list(
    samples: list,
    split_ratios: list,
    shuffle_samples: bool = False,
    seed: Optional[int] = None,
) -> list:
    """Split list into subsets according to given split ratios."""
    if not round(sum(split_ratios), 5) == 1.0:
        raise ValueError("Split ratios don't add up to 1!")
    split_ratios[-1] = 1 - sum(split_ratios[:-1])
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


def split_list_stratify(
    samples: list, sample_labels: list, split_ratios: list, **split_kwargs
) -> list:
    """Stratified split list into subsets according to given split ratios."""

    if not len(sample_labels) == len(samples):
        raise ValueError("'samples' and 'sample_labels' must have the same length")

    unique_labels = list(set(sample_labels))
    n_labels = len(unique_labels)
    label_data = [[] for _ in range(n_labels)]
    for label, value in zip(sample_labels, samples):
        label_data[unique_labels.index(label)].append(value)

    split_label_data = [
        split_list(sublist, split_ratios, **split_kwargs) for sublist in label_data
    ]

    data_split = []
    for i, _ in enumerate(split_ratios):
        data_split.append([item for sublist in split_label_data for item in sublist[i]])

    return data_split


def split_reftimes_cv(
    reftimes: np.ndarray[Any, np.datetime64],
    gap: int = 5,
    interval: int = 360,
    p: list = [0.6, 0.2, 0.2],
    p_tol: float = 0.05,
    uni_tol: float = 0.01,
):
    """Find all possible splits for a given time array and split configuration"""

    if sum(p) != 1:
        raise ValueError("Proportions do not sum up to 1!")

    opt_p = np.array(p[:2]) / (1 - p[2])
    opt_reftimes, test_reftimes = np.array_split(
        reftimes, [int(len(reftimes) * (1 - p[2]))]
    )

    gap = np.array(gap, dtype="timedelta64[D]")
    interval = np.array(interval, dtype="timedelta64[D]")
    n_intervals = int((opt_reftimes[-1] - opt_reftimes[0]) / interval)

    splits = []
    nudge = [-1, 0, 1] if int(n_intervals * opt_p[1]) > 1 else [0, 1]
    for i in nudge:
        which = np.array(
            list(
                itertools.combinations(
                    range(n_intervals), int(n_intervals * opt_p[1] + i)
                )
            ),
            dtype=int,
        )
        splits_ = np.zeros((len(which), n_intervals), dtype=int)
        splits_[np.arange(len(which))[None].T, which] = 1
        splits.append(splits_)

    splits = np.vstack(splits)

    stop_time = time.time() + 20
    good_splits = []
    for choice_array in splits:
        split = pd.Series(-1, index=opt_reftimes, dtype=int)
        t = opt_reftimes[0]
        for idx, choice in enumerate(choice_array):
            next_choice = choice_array[min(idx + 1, len(choice_array) - 1)]
            start = t
            end = t + interval
            split.loc[start:end] = choice
            if next_choice != choice:
                gap_start = end
                gap_end = end + gap
                split.loc[gap_start:gap_end] = 3
            t += interval + gap

        # assign last chunk
        split.loc[opt_reftimes[-1] - gap : opt_reftimes[-1]] = 3
        if check_split(split, opt_p, p_tol, uni_tol):
            good_splits.append(split.values)
        elif len(good_splits) == 30 or time.time() > stop_time:
            break
        else:
            pass

    if len(good_splits) == 0:
        raise RuntimeError("No valid splits were found. Change your parameters!")

    for i in range(len(good_splits)):
        good_splits[i] = np.hstack((good_splits[i], [2] * len(test_reftimes)))

    return good_splits


def check_split(split, p, p_tol, uni_tol):
    """Check that desired criteria are met for a given set"""

    proportion = _check_proportions(split, p, p_tol)
    uniformity = _check_uniformity(split, p, uni_tol)

    return proportion and uniformity


def _check_proportions(split, p, p_tol):
    """Check that proportions are respected between sets"""

    train_times = split[split == 0]
    train_proportion = len(train_times) / len(split)
    if np.isclose(train_proportion, p[0], atol=p_tol):
        return True
    else:
        return False


def _check_uniformity(split, p, uni_tol):
    """Check that reftimes are uniformly distributed in the smaller set"""

    smaller_set_ = np.argmin(p)

    smaller_set = split[split == smaller_set_]
    months = smaller_set.index.month
    month_counts = smaller_set.groupby(months).count()
    dist = month_counts / smaller_set.count()

    uniform_dist = np.ones(12) / 12

    try:
        wsd = wasserstein_distance(dist, uniform_dist)
    except ValueError:
        return False

    if wsd < uni_tol:
        return True
    else:
        return False
