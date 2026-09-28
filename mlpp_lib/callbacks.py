import time

import numpy as np
import scoringrules as sr
import torch
from keras import callbacks


class EnsembleMetrics(callbacks.Callback):
    """Compute a range of probabilistic scores on the validation data at the end of
    each epoch, from `n_samples` samples of the predicted distribution (for the
    first target variable only).

    The CRPS is computed with the "qd" estimator, which gives the same values as the
    (biased) energy form used by mlpp-lib < 1.0, with a cost growing linearly with
    `n_samples` (up to sorting). Predictions are computed in batches of `batch_size`.
    """

    def __init__(self, n_samples=50, thresholds=None, batch_size=100_000):
        super().__init__()
        self.n_samples = n_samples
        self.thresholds = thresholds or []
        self.batch_size = batch_size

    def add_validation_data(self, validation_data) -> None:
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs):
        """Compute a range of probabilistic scores at the end of each epoch."""
        y_pred = []
        with torch.no_grad():
            for start in range(0, len(self.X_val), self.batch_size):
                x = self.X_val[start : start + self.batch_size]
                samples = self.model(x).sample((self.n_samples,))
                y_pred.append(samples[:, :, 0].cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=1).T
        y_val = np.squeeze(self.y_val)
        assert y_val.shape[0] == y_pred.shape[0]
        assert y_pred.shape[1] == self.n_samples

        def exceedances(x, thr):
            exceeds = (x > thr).astype(float)
            exceeds[np.where(np.isnan(x))] = np.nan
            return exceeds

        logs["val_ensstd"] = float(np.std(y_pred, axis=1).mean())
        logs["val_crps"] = float(
            np.nanmean(sr.crps_ensemble(y_val, y_pred, m_axis=1, estimator="qd"))
        )
        for thr in self.thresholds:
            y_val_thr = np.maximum(y_val, thr)
            y_pred_thr = np.maximum(y_pred, thr)
            logs[f"val_crps_{thr}"] = float(
                np.nanmean(
                    sr.crps_ensemble(y_val_thr, y_pred_thr, m_axis=1, estimator="qd")
                )
            )
            y_val_bin = exceedances(y_val, thr)
            y_pred_prob = exceedances(y_pred, thr).mean(axis=1)
            logs[f"val_bs_{thr}"] = float(
                np.nanmean(sr.brier_score(y_val_bin, y_pred_prob))
            )


class TimeHistory(callbacks.Callback):
    """Callback to log epoch run times"""

    def on_epoch_begin(self, *args):
        self.epoch_time_start = time.monotonic()

    def on_epoch_end(self, epoch, logs):
        logs["epoch_time"] = time.monotonic() - self.epoch_time_start
