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
