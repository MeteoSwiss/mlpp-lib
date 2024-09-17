import numpy as np
import numpy as np
import xarray as xr

import mlpp_lib.ensemble as ens


def test_sortby():
    """For a standard case, sortby should return the same results as numpy sort."""
    test_ds = xr.DataArray(np.random.random((3, 3, 3)), dims=("a", "b", "c"))
    out = ens.sortby(test_ds, "a")
    assert isinstance(out, xr.DataArray)
    assert out.dims[0] == "rank"
    ref = np.sort(test_ds, axis=0)
    assert np.allclose(out, ref)


def test_equidistant_resampling():
    test_ds = xr.DataArray(
        np.random.random((3, 20)), dims=("a", "realization")
    ).to_dataset(name="var")
    out = ens.equidistant_resampling(test_ds, 5)
    assert isinstance(out, xr.Dataset)
    assert out.sizes["realization"] == 5


def test_equidistant_resampling_circular():
    test_ds = xr.DataArray(
        np.random.randint(0, 360, (3, 20)), dims=("a", "realization")
    ).to_dataset(name="var_direction")
    out = ens.equidistant_resampling(test_ds, 5)
    assert isinstance(out, xr.Dataset)
    assert out.sizes["realization"] == 5


def test_compute_ecc():
    test_ds = xr.DataArray(
        np.random.random((3, 10)), dims=("a", "realization")
    ).to_dataset(name="var")
    test_template = xr.DataArray(
        np.random.randint(1, 5, (3, 5)), dims=("a", "realization")
    )
    out = ens.compute_ecc(test_ds, test_template)
    assert isinstance(out, xr.Dataset)
    assert out.sizes["realization"] == 5
