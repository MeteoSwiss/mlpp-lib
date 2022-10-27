import pytest
import xarray as xr

from mlpp_lib.datasets import get_tensor_dataset, split_dataset


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


def test_split_dataset(features_dataset):
    """"""
    splits = dict(
        a=dict(station=["AAA", "BBB"]),
        b=dict(t=slice(0, 1)),
        c=dict(t=slice(0, 1), station=["AAA", "BBB"]),
    )
    out = split_dataset(features_dataset, splits)
    dims_in = features_dataset.dims
    assert list(out.keys()) == ["a", "b", "c"]
    assert out["a"].dims == dict(dims_in, station=2)
    assert out["b"].dims == dict(dims_in, t=2)
    assert out["c"].dims == dict(dims_in, t=2, station=2)


def test_split_dataset_missing_dims(features_dataset):
    """"""
    splits = dict(
        a=dict(station=["AAA", "BBB"], asd=[1, 2, 3]),
    )
    with pytest.raises(KeyError):
        out = split_dataset(features_dataset, splits, ignore_missing_dims=False)
    out = split_dataset(features_dataset, splits, ignore_missing_dims=True)
    assert list(out.keys()) == ["a"]
