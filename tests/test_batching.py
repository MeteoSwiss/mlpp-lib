from itertools import product

import numpy as np
import pytest

from mlpp_lib.batching import get_tensor_dataset


@pytest.mark.parametrize("event_dims,", [[], ["t"]])
def test_get_tensor_dataset(features_dataset, targets_dataset, event_dims):

    tensor_dataset = get_tensor_dataset(features_dataset, targets_dataset, event_dims)

    event_dims_size = [features_dataset.dims.mapping[dim] for dim in event_dims]
    expected_batch_x_shape = (None, *event_dims_size, len(features_dataset.data_vars))
    expected_batch_y_shape = (None, *event_dims_size, len(targets_dataset.data_vars))

    x_shape = tensor_dataset[0].shape
    y_shape = tensor_dataset[1].shape

    assert x_shape[1:] == expected_batch_x_shape[1:]
    assert y_shape[1:] == expected_batch_y_shape[1:]
