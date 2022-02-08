import numpy as np

from mlpp_lib import generators as gen


def test_get_batches_list(features_dataset, stack_dims):

    batches_list = gen.get_batches_list(features_dataset, stack_dims)

    assert len(batches_list) == 45
    assert list(batches_list[0].keys()) == list(stack_dims.keys())


def test_get_batches_shapes(features_dataset, targets_dataset, stack_dims) -> None:

    batch_x_shape, batch_y_shape = gen.get_batches_shapes(
        features_dataset, targets_dataset, stack_dims
    )

    assert batch_x_shape == (None, 24, 4)
    assert batch_y_shape == (None, 24, 2)


def test_batch_generator_dataset(features_dataset, targets_dataset, stack_dims) -> None:

    dataset = gen.BatchGeneratorDataset(features_dataset, targets_dataset, stack_dims)

    sample_batches = dataset.take(1).get_single_element()

    np.testing.assert_equal(
        sample_batches[0][0][0].numpy(),
        np.array([0.34558418, 0.82161814, 0.33043706, -1.3031572], dtype="float32"),
    )

    np.testing.assert_equal(
        sample_batches[1][0][0].numpy(),
        np.array([0.34558419, 0.82161814], dtype="float32"),
    )


def test_batch_generator_sequence(
    features_dataset, targets_dataset, stack_dims
) -> None:

    sequence = gen.BatchGeneratorSequence(features_dataset, targets_dataset, stack_dims)

    np.testing.assert_equal(
        sequence[0][0][0][0].astype("float32"),
        np.array([0.34558418, 0.82161814, 0.33043706, -1.3031572], dtype="float32"),
    )

    np.testing.assert_equal(
        sequence[0][1][0][0].astype("float32"),
        np.array([0.34558419, 0.82161814], dtype="float32"),
    )
