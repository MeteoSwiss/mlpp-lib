from typing import Optional
from typing_extensions import Self

import numpy as np
import pandas as pd


class RegressionResampler:
    """
    Resample a regression dataset to adjust for imbalanced target distributions.

    This class computes sampling probabilities inversely proportional to the frequency
    of the target variable (either directly or using bins) and allows resampling
    input-target pairs accordingly.
    """

    def __init__(self, prob: np.ndarray):
        """
        Initialize the resampler with sampling probabilities.

        Parameters
        ----------
        prob : np.ndarray
            Array of sampling probabilities for each sample.
        """
        self.prob = prob

    @classmethod
    def fit_resample(
        cls: Self,
        *arrays: np.ndarray,
        y: np.ndarray,
        n_bins: Optional[int] = None,
        size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit resampling probabilities and return resampled (x, y) pairs.

        Parameters
        ----------
        *arrays : np.ndarray
            Input arrays to be resampled (e.g., x, z, etc.), each with shape (n_samples, ...).
        y : np.ndarray
            Target values, shape (n_samples,).
        n_bins : int, optional
            If given, the target variable is binned into `n_bins` bins before computing frequencies.
        size : int, optional
            Number of samples to draw. If None, the original number of samples is used.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple of np.ndarray
            Tuple of resampled arrays in the same order as provided (including `y`).
        """
        if n_bins is not None and n_bins <= 0:
            raise ValueError("Resample n_bins must be a positive integer.")
        if size is not None and size <= 0:
            raise ValueError("Resample size must be a positive integer.")
        resampler = cls.fit(y, n_bins)
        return resampler.resample(*arrays, y, size=size, random_seed=random_seed)

    @classmethod
    def fit(cls: Self, y: np.ndarray, n_bins: Optional[int] = None) -> Self:
        """
        Compute sampling probabilities based on the target distribution.

        Parameters
        ----------
        y : np.ndarray
            Target values, shape (n_samples,).
        n_bins : int, optional
            If given, the values in `y` are grouped into `n_bins` bins before computing frequency.

        Returns
        -------
        RegressionResampler
            Instance initialized with sampling probabilities.

        Raises
        ------
        ValueError
            If `y` is not 1-dimensional.
        """
        y = y.squeeze()
        if n_bins:
            bins = pd.cut(y, bins=n_bins, precision=1)
            if bins.isna().any():
                raise ValueError(
                    "NaNs encountered during binning. Check the input `y`."
                )
            labels = bins.codes
        else:
            labels = y
        labels_freq = 1 / pd.Series(labels).value_counts()
        prob = np.vectorize(labels_freq.to_dict().get)(labels)
        prob /= prob.sum()
        return cls(prob)

    def sample_indices(
        self,
        size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate sample indices using the computed resampling probabilities.

        Parameters
        ----------
        size : int, optional
            Number of indices to sample. If None, defaults to the original data size.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Array of sampled indices.
        """
        size = size or len(self.prob)
        np.random.seed(random_seed)
        return np.random.choice(
            range(len(self.prob)), size=size, p=self.prob, replace=True
        )

    def resample(
        self,
        *arrays: np.ndarray,
        size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resample the input and target arrays using computed probabilities.

        Parameters
        ----------
        *arrays : np.ndarray
            Input arrays to resample. Each array must have the same first dimension length.
        size : int, optional
            Number of samples to draw. If None, the original number of samples is used.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple of np.ndarray
            Tuple of resampled arrays, in the same order as given.

        Raises
        ------
        AssertionError
            If any input array does not match the length of the sampling probabilities.

        """
        for array in arrays:
            assert array.shape[0] == len(
                self.prob
            ), "All input arrays must align with prob length."

        indices = self.sample_indices(size=size, random_seed=random_seed)
        return tuple(array[indices] for array in arrays)
