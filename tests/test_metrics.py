import pytest
import numpy as np
import keras.ops as ops
from mlpp_lib import metrics


def test_bias():
    y_true = ops.convert_to_tensor([1, 2, 3, 4, 5])
    y_pred = ops.convert_to_tensor([0.8, 2.2, 2.9, 4.1, 5.5])
    result = metrics.expected_bias(y_true, y_pred)
    expected_result = (-0.2 + 0.2 - 0.1 + 0.1 + 0.5) / 5
    assert result.numpy() == pytest.approx(expected_result)

def test_mae():
    y_true = ops.convert_to_tensor([1, 2, 3, 4, 5])
    y_pred = ops.convert_to_tensor([0.8, 2.2, 2.9, 4.1, 5.5])
    result = metrics.expected_mean_absolute_error(y_true, y_pred)
    
    expected_result = (0.2 + 0.2 + 0.1 + 0.1 + 0.5) / 5
    assert result.numpy() == pytest.approx(expected_result)

class TestMAEBusts:
    @pytest.fixture
    def maebusts(self):
        return metrics.MAEBusts(threshold=0.5)

    def test_maebusts_initialization(self, maebusts):
        assert maebusts.threshold == 0.5
        assert maebusts.n_busts.numpy() == 0
        assert maebusts.n_samples.numpy() == 0

    def test_maebusts_update_state(self, maebusts):
        y_true = ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
        y_pred = ops.convert_to_tensor([0.8, 2.2, 2.5, 4.5, 4.4, 6.6])
        maebusts.update_state(y_true, y_pred)
        assert maebusts.n_busts.numpy() == 2
        assert maebusts.n_samples.numpy() == 6
        maebusts.reset_state()
        sample_weight = ops.convert_to_tensor([1, 0, 1, 0, 1, 0])
        maebusts.update_state(y_true, y_pred, sample_weight)
        assert maebusts.n_busts.numpy() == 1
        assert maebusts.n_samples.numpy() == 3
        maebusts.reset_state()
        assert maebusts.n_busts.numpy() == 0
        assert maebusts.n_samples.numpy() == 0

    def test_maebusts_result(self, maebusts):
        y_true = ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
        y_pred = ops.convert_to_tensor([0.8, 2.2, 2.5, 4.5, 4.4, 6.6])
        maebusts.update_state(y_true, y_pred)
        assert maebusts.result().numpy() == pytest.approx(2 / 6)
        maebusts.reset_state()
        sample_weight = ops.convert_to_tensor([1, 0, 1, 0, 1, 0])
        maebusts.update_state(y_true, y_pred, sample_weight)
        assert maebusts.result().numpy() == pytest.approx(1 / 3)
        maebusts.reset_state()
        assert np.isnan(maebusts.result().numpy())
