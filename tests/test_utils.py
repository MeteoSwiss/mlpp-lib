import xarray as xr


from mlpp_lib import utils


def test_as_weather():
    ds_in = xr.Dataset(
        {
            "wind_speed": ("x", [0, 1, 2]),
            "source:wind_speed": ("x", [0, 1, 2]),
            "asd": ("x", [0, 1, 2]),
        }
    )
    ds_out = utils.as_weather(ds_in)
    xr.testing.assert_equal(ds_out, ds_in)

    ds_in = xr.Dataset(
        {
            "cos_wind_from_direction": ("x", [0, 1, 2]),
            "sin_wind_from_direction": ("x", [0, 1, 2]),
        }
    )
    ds_out = utils.as_weather(ds_in)
    assert "wind_from_direction" in ds_out

    ds_in = xr.Dataset(
        {
            "northward_wind": ("x", [0, 1, 2]),
            "eastward_wind": ("x", [0, 1, 2]),
        }
    )
    ds_out = utils.as_weather(ds_in)
    assert "wind_from_direction" in ds_out
    assert "wind_speed" in ds_out


import keras
import pytest

from mlpp_lib import losses, metrics


@pytest.mark.parametrize(
    "config, expected",
    [
        ("CRPSNormal", losses.CRPSNormal),
        ({"CRPSEnsemble": {"num_samples": 7}}, losses.CRPSEnsemble),
        ({"CRPSEnsemble": None}, losses.CRPSEnsemble),
        (
            {"DistributionLossWrapper": "scoringrules.crps_normal"},
            losses.DistributionLossWrapper,
        ),
        (
            {"SampleLossWrapper": {"scoringrules.crps_ensemble": {"num_samples": 7}}},
            losses.SampleLossWrapper,
        ),
        (
            {"SampleLossWrapper": {"fn": "scoringrules.twcrps_ensemble", "a": 0.1}},
            losses.SampleLossWrapper,
        ),
        ({"MeanSquaredError": {}}, keras.losses.MeanSquaredError),
    ],
)
def test_get_loss(config, expected):
    assert isinstance(utils.get_loss(config), expected)


def test_get_loss_special_cases():
    assert utils.get_loss({"CRPSEnsemble": {"num_samples": 7}}).num_samples == 7
    assert callable(utils.get_loss("mse"))
    with pytest.warns(DeprecationWarning, match="crps_energy"):
        loss = utils.get_loss("crps_energy")
    assert isinstance(loss, losses.CRPSEnsemble)
    assert loss.num_samples == 1000
    with pytest.raises(KeyError):
        utils.get_loss("not_a_loss")


def test_get_metric():
    assert utils.get_metric("bias") is metrics.bias
    assert isinstance(
        utils.get_metric({"MAEBusts": {"threshold": 1}}), metrics.MAEBusts
    )
    assert callable(utils.get_metric("mae"))
    with pytest.raises(KeyError):
        utils.get_metric("not_a_metric")


def test_get_model_errors():
    with pytest.raises(KeyError):
        utils.get_model((3,), 1, {"not_a_model": {}})


def test_process_out_bias_init():
    import numpy as np

    y = np.array([[1.0, 10.0], [3.0, np.nan], [2.0, 20.0]])
    np.testing.assert_allclose(utils.process_out_bias_init(y, "mean", []), [2.0, 15.0])
    assert utils.process_out_bias_init(y, None, []) == "zeros"
    assert utils.process_out_bias_init(y, "ones", []) == "ones"
