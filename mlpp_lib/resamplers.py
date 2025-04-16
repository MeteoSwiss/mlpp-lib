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
        x: np.ndarray,
        y: np.ndarray,
        n_bins: Optional[int] = None,
        size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit resampling probabilities and return resampled (x, y) pairs.

        Parameters
        ----------
        x : np.ndarray
            Input features, shape (n_samples, ...).
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
            Resampled (x, y) arrays.
        """
        if size is not None and size <= 0:
            raise ValueError("Resample size must be a positive integer.")
        return cls.fit(y, n_bins).resample(x, y, size, random_seed)

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
        if y.ndim > 1:
            raise ValueError("Only 1D arrays are supported.")
        if n_bins:
            bins = pd.cut(y, bins=n_bins, precision=1)
            if bins.isna().any():
                raise ValueError(
                    "NaNs encountered during binning. Check the input `y`."
                )
            labels = bins.codes
        else:
            labels = y
        labels_freq = 1 / pd.value_counts(labels)
        prob = np.vectorize(labels_freq.to_dict().get)(labels)
        prob /= prob.sum()
        return cls(prob)

    def resample(
        self,
        x: np.ndarray,
        y: np.ndarray,
        size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resample the input and target arrays using computed probabilities.

        Parameters
        ----------
        x : np.ndarray
            Input features, shape (n_samples, ...).
        y : np.ndarray
            Target values, shape (n_samples,).
        size : int, optional
            Number of samples to draw. If None, the original number of samples is used.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple of np.ndarray
            Resampled (x, y) arrays.
        """
        assert x.shape[0] == len(self.prob)
        assert y.shape[0] == len(self.prob)
        size = size or x.shape[0]
        np.random.seed(random_seed)
        new_indices = np.random.choice(
            range(len(y)), size=size, p=self.prob, replace=True
        )
        return x[new_indices], y[new_indices]
