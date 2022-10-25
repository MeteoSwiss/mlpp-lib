import pytest
import xarray as xr

from mlpp_lib.batching import get_tensor_dataset


@pytest.mark.parametrize("event_dims,", [[], ["t"]])
def test_get_tensor_dataset(features_dataset, targets_dataset, event_dims):

    tensors = get_tensor_dataset(
        features_dataset, targets_dataset, event_dims=event_dims
    )

    assert all([isinstance(tensor, xr.DataArray) for tensor in tensors])

    event_dims_size = [features_dataset.dims.mapping[dim] for dim in event_dims]
    expected_batch_x_shape = (None, *event_dims_size, len(features_dataset.data_vars))
    expected_batch_y_shape = (None, *event_dims_size, len(targets_dataset.data_vars))

    x_shape = tensors[0].shape
    y_shape = tensors[1].shape

    assert x_shape[1:] == expected_batch_x_shape[1:]
    assert y_shape[1:] == expected_batch_y_shape[1:]


def test_get_tensor_dataset_with_None(features_dataset, targets_dataset):

    tensor_dataset = get_tensor_dataset(
        features_dataset, None, targets_dataset, event_dims=[]
    )
    assert tensor_dataset[1] is None
