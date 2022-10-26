import numpy as np

from mlpp_lib.datasets import split_dataset
from mlpp_lib.standardizers import standardize_split_dataset


def test_standardize_split_dataset(features_dataset):
    splits = dict(
        train=dict(station=[chr(i) * 3 for i in range(ord("A"), ord("M"))]),
        val=dict(station=[chr(i) * 3 for i in range(ord("N"), ord("Z"))]),
    )
    features_dataset["coe:x1"][0, 0, 0] = np.nan
    dataset_split = split_dataset(features_dataset, splits)
    dataset_split = standardize_split_dataset(dataset_split)
    assert dataset_split["train"]["coe:x1"][0, 0, 0] == -5
