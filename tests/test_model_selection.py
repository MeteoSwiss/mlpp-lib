import numpy as np

import mlpp_lib.model_selection as ms


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
