import time

import numpy as np
import scoringrules as sr
from keras import callbacks


class EnsembleMetrics(callbacks.Callback):
    def __init__(self, n_samples=50, thresholds=None):
        super(callbacks.Callback, self).__init__()
        self.n_samples = n_samples
        self.thresholds = thresholds or []

    def add_validation_data(self, validation_data) -> None:
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs):
        """Compute a range of probabilistic scores at the end of each epoch."""
        y_pred = self.model(self.X_val).sample((self.n_samples,))

        y_pred = y_pred.numpy()[:, :, 0].T
        y_val = np.squeeze(self.y_val)
        assert y_val.shape[0] == y_pred.shape[0]
        assert y_pred.shape[1] == self.n_samples

        def exceedances(x, thr):
            exceeds = (x > thr).astype(float)
            exceeds[np.where(np.isnan(x))] = np.nan
            return exceeds

        logs["val_ensstd"] = np.std(y_pred, axis=1).mean().astype(float)
        logs["val_crps"] = sr.crps_ensemble(y_val, y_pred, m_axis=1).mean()
        for thr in self.thresholds:
            y_val_thr = np.maximum(y_val, thr)
            y_pred_thr = np.maximum(y_pred, thr)
            logs[f"val_crps_{thr}"] = sr.crps_ensemble(
                y_val_thr, y_pred_thr, m_axis=1
            ).mean()
            y_val_bin = exceedances(y_val, thr)
            y_pred_prob = exceedances(y_pred, thr).mean(axis=1)
            logs[f"val_bs_{thr}"] = sr.brier_score(y_val_bin, y_pred_prob).mean()


class TimeHistory(callbacks.Callback):
    """Callback to log epoch run times"""

    def on_epoch_begin(self, *args):
        self.epoch_time_start = time.monotonic()

    def on_epoch_end(self, epoch, logs):
        logs["epoch_time"] = time.monotonic() - self.epoch_time_start
