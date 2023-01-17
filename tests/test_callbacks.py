import numpy as np
import pytest

from mlpp_lib import callbacks


class TestCallbacks(object):
    @pytest.fixture(autouse=True)
    def init_model(self, get_prob_model):
        n_features = 5
        n_samples = 1000
        x_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randn(n_samples, 1)
        x_val = np.random.randn(int(n_samples * 0.2), n_features)
        y_val = np.random.randn(int(n_samples * 0.2), 1)
        self.train_data = (x_train, y_train)
        self.validation_data = (x_val, y_val)
        self.model = get_prob_model(n_features, 1)

    def _train_with_callback(self, custom_callback):
        return self.model.fit(
            *self.train_data,
            validation_data=self.validation_data,
            batch_size=128,
            epochs=1,
            verbose=0,
            callbacks=[custom_callback],
        )

    def test_ProperScores(self):
        custom_callback = callbacks.ProperScores(thresholds=[0, 1])
        custom_callback.add_validation_data(self.validation_data)
        res = self._train_with_callback(custom_callback)
        assert "val_crps" in res.history
        assert "val_crps_0" in res.history
        assert "val_crps_1" in res.history
        assert "val_bs_0" in res.history
        assert "val_bs_1" in res.history

    def test_TimeHistory(self):
        custom_callback = callbacks.TimeHistory()
        res = self._train_with_callback(custom_callback)
        assert "time" in res.history
