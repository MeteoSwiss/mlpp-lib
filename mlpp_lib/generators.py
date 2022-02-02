import itertools
from dataclasses import dataclass, field
from functools import partial
from typing import Union
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf


@dataclass
class BatchGeneratorSequence(tf.keras.utils.Sequence):
    """
    Generate custom batches with desired dimensions and sizes.
    Subclasses the Sequence utility from Keras.
    """

    features: xr.Dataset = field(repr=False)
    targets: xr.Dataset = field(repr=False)
    stack_dims: dict[str, int]
    shuffle: bool = True

    x_shape: tuple[int] = field(init=False)
    y_shape: tuple[int] = field(init=False)

    def __post_init__(self) -> None:
        self.x_shape, self.y_shape = get_batches_shapes(
            self.features, self.targets, self.stack_dims
        )
        self.batches_idx_list = get_batches_list(self.features, self.stack_dims)

    def __getitem__(self, idx: int) -> tuple[np.ndarray]:
        """Return a single batch"""
        return select_stack_batch(
            self.features, self.targets, self.stack_dims, self.batches_idx_list[idx]
        )

    def __len__(self) -> int:
        """Return the number of batches created for the input dataset"""
        return len(self.batches_idx_list)

    def on_epoch_end(self) -> None:
        """Shuffle the list of batches at the end of an epoch"""
        if self.shuffle:
            np.random.shuffle(self.batches_idx_list)


class BatchGeneratorDataset(tf.data.Dataset):
    """
    Generate custom batches with desired dimensions and sizes.
    Subclasses the Dataset utility from tf.data.
    """

    def _generator(
        features: xr.Dataset,
        targets: xr.Dataset,
        stack_dims: dict[str, int],
        batches: list[dict[str, Union[int, str, datetime, float]]],
    ) -> tuple[np.ndarray]:
        for batch in batches:
            yield select_stack_batch(features, targets, stack_dims, batch)

    def __new__(
        cls, features: xr.Dataset, targets: xr.Dataset, stack_dims: dict[str, int]
    ) -> tf.data.Dataset:

        x_shape, y_shape = get_batches_shapes(features, targets, stack_dims)
        batches_idx_list = get_batches_list(features, stack_dims)
        generator = partial(
            cls._generator, features, targets, stack_dims, batches_idx_list
        )

        OUTPUT_SIGNATURE = (
            tf.TensorSpec(shape=x_shape, dtype=tf.float32),
            tf.TensorSpec(shape=y_shape, dtype=tf.float32),
        )

        return tf.data.Dataset.from_generator(
            generator, output_signature=OUTPUT_SIGNATURE
        )


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from a list"""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_batches_list(
    features: xr.Dataset, stack_dims: dict[str, int]
) -> list[dict[str, Union[int, str, datetime, float]]]:
    """Return a list of dictionaries that can be used for selecting batches with xarray objects"""
    stack_batches = [
        [b for b in chunks(features[dim].values, size) if len(b) == size]
        for dim, size in stack_dims.items()
    ]
    batches = list(itertools.product(*stack_batches))
    batches = [{d: b[i] for i, d in enumerate(stack_dims)} for b in batches]
    return batches


def select_stack_batch(
    features: xr.Dataset,
    targets: xr.Dataset,
    stack_dims: dict[str, int],
    batch: dict[str, Union[int, str, datetime, float]],
) -> tuple[np.ndarray]:

    batch_x = (
        features.loc[batch]
        .to_array("variable")
        .stack(sample=stack_dims)
        .transpose("sample", ..., "variable")
    )
    batch_y = (
        targets.loc[batch]
        .to_array("target")
        .stack(sample=stack_dims)
        .transpose("sample", ..., "target")
    )

    mask_x_dims = [dim for dim in batch_x.dims if dim != "sample"]
    mask_y_dims = [dim for dim in batch_y.dims if dim != "sample"]

    batch_x = batch_x[np.isfinite(batch_y).all(dim=mask_y_dims)]
    batch_y = batch_y[np.isfinite(batch_y).all(dim=mask_y_dims)]
    batch_y = batch_y[np.isfinite(batch_x).all(dim=mask_x_dims)]
    batch_x = batch_x[np.isfinite(batch_x).all(dim=mask_x_dims)]

    return (batch_x.values, batch_y.values)


def get_batches_shapes(
    features: xr.Dataset, targets: xr.Dataset, stack_dims: dict[str, int]
) -> tuple[tuple]:
    remaining_dims = list(set(features.dims.mapping) - set(stack_dims.keys()))
    remaining_dims_size = [features.dims.mapping[dim] for dim in remaining_dims]
    batch_x_shape = (None, *remaining_dims_size, len(features.data_vars))
    batch_y_shape = (None, *remaining_dims_size, len(targets.data_vars))
    return batch_x_shape, batch_y_shape
