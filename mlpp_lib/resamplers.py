from typing import Optional
from typing_extensions import Self

import numpy as np
import pandas as pd


class RegressionResampler:
    def __init__(self, prob: np.array):
        self.prob = prob

    @classmethod
    def fit_resample(
        cls: Self,
        x: np.array,
        y: np.ndarray,
        n_bins: Optional[int] = None,
        size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> tuple[np.array, np.array]:
        """Compute resampling probabilities and run resampling."""
        return cls.fit(y, n_bins).resample(x, y, size, random_seed)

    @classmethod
    def fit(cls: Self, y: np.ndarray, n_bins: Optional[int] = None) -> Self:
        """Compute resampling probabilities based on sample frequency."""
        y = y.squeeze()
        if y.ndim > 1:
            raise ValueError("We can only fit to 1-D array")
        if n_bins:
            bins = pd.cut(y, bins=n_bins, precision=1)
            labels = bins.codes
        else:
            labels = y
        labels_freq = 1 / pd.value_counts(labels)
        prob = np.vectorize(labels_freq.to_dict().get)(labels)
        prob /= prob.sum()
        return cls(prob)

    def resample(
        self,
        x: np.array,
        y: np.array,
        size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> tuple[np.array, np.array]:
        """Run resampling."""
        assert x.shape[0] == len(self.prob)
        assert y.shape[0] == len(self.prob)
        size = size or x.shape[0]
        np.random.seed(random_seed)
        new_indices = np.random.choice(
            range(len(y)), size=size, p=self.prob, replace=True
        )
        return x[new_indices], y[new_indices]
