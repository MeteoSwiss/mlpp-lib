from itertools import product

import numpy as np
import pytest

from mlpp_lib.batching import get_tensor_dataset


@pytest.mark.parametrize("event_dims,", ([], ["t"]))
def test_get_tensor_dataset(features_dataset, targets_dataset, event_dims):

    batch_size = 32
    batched_dataset = get_tensor_dataset(
        features_dataset, targets_dataset, event_dims, batch_size
    )

    event_dims_size = [features_dataset.dims.mapping[dim] for dim in event_dims]
    expected_batch_x_shape = (None, *event_dims_size, len(features_dataset.data_vars))
    expected_batch_y_shape = (None, *event_dims_size, len(targets_dataset.data_vars))

    element_spec = batched_dataset.element_spec
    batch_x_shape = tuple(element_spec[0].shape)
    batch_y_shape = tuple(element_spec[1].shape)

    assert batch_x_shape == expected_batch_x_shape
    assert batch_y_shape == expected_batch_y_shape
