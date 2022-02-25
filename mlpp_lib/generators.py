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

    Parameters
    ----------
    features: xr.Dataset
        Dataset containing features predictors
    targets: xr.Dataset
        Dataset containing targets predictands.
    stack_dims: dict
        Dimensions that are going to be stacked
        and the number of samples that will be included in a batch.
    shuffle: bool
        Whether or not to shuffle batches at the end of each epoch.

    Attributes
    ----------
    x_shape: tuple
        Shape of predictors, with batch size (None) as first dimension.
    y_shape: tuple
        Shape of predictands, with batch size (None) as first dimension.

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

    Parameters
    ----------
    features: xr.Dataset
        Dataset containing features predictors
    targets: xr.Dataset
        Dataset containing targets predictands.
    stack_dims: dict
        Dimensions that are going to be stacked
        and the number of samples that will be included in a batch.

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
    """
    Return a list of dictionaries for indexing xarray objects.

    Parameters
    ----------
    features: xr.Dataset
        Dataset containing input features.
    stack_dims: dict
        Dimensions that are going to be stacked
        and the number of samples that will be included in a batch.

    Returns
    -------
    batches: list
        List of dict-like indexes for xarray objects. Each element of the list
        can be used to select a batch from the Datasets.
    """

    stack_batches = []
    for dim, size in stack_dims.items():
        size = min(features.sizes[dim], size)
        chunked = chunks(features[dim].values, size)
        stack_batches.append([b for b in chunked if len(b) == size])

    batches = list(itertools.product(*stack_batches))
    batches = [{d: b[i] for i, d in enumerate(stack_dims)} for b in batches]
    return batches


def select_stack_batch(
    features: xr.Dataset,
    targets: xr.Dataset,
    stack_dims: dict[str, int],
    batch: dict[str, Union[int, str, datetime, float]],
) -> tuple[np.ndarray]:
    """
    Construct a batch.

    Selects samples from the provided datasets and stack dimensions to form a batch,
    then removes samples with missing values.

    Parameters
    ----------
    features: xr.Dataset
        Dataset containing input features.
    targets: xr.Dataset
        Dataset containing output targets.
    stack_dims: dict
        Dimensions that are going to be stacked
        and the number of samples that will be included in a batch.
    batch: dict
        A dict-like indexer for xarray objects.

    Returns
    -------
    batch_x: np.ndarray
        Predictors batch.
    batch_y: np.ndarray
        Predictands batch.
    """

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
    """
    Return the shapes of predictors and predictands batches.

    Parameters
    ----------
    features: xr.Dataset
        Dataset containing features predictors.
    targets: xr.Dataset
        Dataset containing targets predictands.
    stack_dims: dict
        Dimensions that are going to be stacked
        and the number of samples that will be included in a batch.

    Returns
    -------
    batch_x_shape: tuple
        Shape of predictors, with batch size (None) as first dimension.
    batch_y_shape: tuple
        Shape of predictands, with batch size (None) as first dimension.

    """
    remaining_dims = list(set(features.dims.mapping) - set(stack_dims.keys()))
    remaining_dims_size = [features.dims.mapping[dim] for dim in remaining_dims]
    batch_x_shape = (None, *remaining_dims_size, len(features.data_vars))
    batch_y_shape = (None, *remaining_dims_size, len(targets.data_vars))
    return batch_x_shape, batch_y_shape
