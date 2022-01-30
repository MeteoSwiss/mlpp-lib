import numpy as np

from mlpp_lib.data_handling import split_list, split_list_stratify


def get_split_lengths(split_ratios, n):
    split_lengths = []
    start = 0
    for ratio in split_ratios:
        end = start + ratio
        split_lengths.append(int(end * n) - int(start * n))
        start = end
    return split_lengths


def test_list_split():
    """"""
    n_samples = 101
    split_ratios = [0.3, 0.6, 0.1]

    samples = list(range(n_samples))
    sample_split = split_list(samples, split_ratios)
    split_lengths = get_split_lengths(split_ratios, n_samples)
    for i in range(len(split_ratios)):
        assert len(sample_split[i]) == split_lengths[i]

    tmp_data = []
    for i in range(len(split_ratios)):
        tmp_data += sample_split[i]
    assert len(tmp_data) == len(samples)

    sample_split_shuffled = split_list(samples, split_ratios, shuffle_samples=True)
    assert sample_split_shuffled != sample_split


def test_split_list_stratify():
    """"""
    n_samples = 101
    split_ratios = [0.5, 0.2, 0.3]
    n_labels = 7

    samples = list(range(n_samples))
    labels = np.random.randint(0, n_labels, n_samples)

    sample_split = split_list_stratify(samples, labels, split_ratios)
    assert len(sample_split) == len(split_ratios)
    assert len([item for sublist in sample_split for item in sublist]) == n_samples

    for label in set(labels):
        subdata = [s for l, s in zip(labels, samples) if l == label]
        split_lengths = get_split_lengths(split_ratios, len(subdata))
        assert sum([s in sample_split[0] for s in subdata]) == split_lengths[0]
        assert sum([s in sample_split[1] for s in subdata]) == split_lengths[1]
        assert sum([s in sample_split[2] for s in subdata]) == split_lengths[2]
