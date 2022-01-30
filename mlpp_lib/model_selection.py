import random
from typing import Optional


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
