# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import importlib.util

import numpy as np
import pytest

import spectrochempy.core.dataset.nddataset as ndmodule
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

if importlib.util.find_spec("xarray") is not None:
    import xarray as xr
else:  # pragma: no cover - depends on optional dependency availability
    xr = None


def _make_dataset(data=None, *, complex_data=False):
    if data is None:
        if complex_data:
            data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
        else:
            data = np.array([[1.0, 2.0], [3.0, 4.0]])

    return NDDataset(
        data,
        dims=["y", "x"],
        coordset=[
            Coord([10.0, 20.0], name="y", title="time", units="s"),
            Coord([1000.0, 1200.0], name="x", title="wavenumber", units="cm^-1"),
        ],
        units="volt",
        mask=np.array([[False, True], [False, False]]),
        title="IR spectra",
        name="spectra",
        meta={
            "sample": "demo",
            "nested": {"temperature": 298.15, "labels": ["a", "b"]},
            "count": 2,
        },
    )


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_simple_float_dataset(tmp_path):
    ds = _make_dataset()
    filename = tmp_path / "dataset.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    assert np.array_equal(rebuilt.data, ds.data)
    assert tuple(rebuilt.dims) == tuple(ds.dims)


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_coordinates_units_mask_and_metadata(tmp_path):
    ds = _make_dataset()
    filename = tmp_path / "dataset.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    assert np.array_equal(rebuilt.coord("x").data, ds.coord("x").data)
    assert np.array_equal(rebuilt.coord("y").data, ds.coord("y").data)
    assert str(rebuilt.coord("x").units) == str(ds.coord("x").units)
    assert str(rebuilt.units) == str(ds.units)
    assert np.array_equal(rebuilt.mask, ds.mask)
    assert rebuilt.meta["sample"] == "demo"
    assert rebuilt.meta["nested"]["temperature"] == 298.15
    assert rebuilt.name == ds.name
    assert rebuilt.title == ds.title


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_complex_roundtrip_uses_split_real_imag_convention(tmp_path):
    ds = _make_dataset(complex_data=True)
    filename = tmp_path / "complex.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    with xr.open_dataset(filename, engine="scipy") as opened:
        assert opened.attrs["scpy_complex_representation"] == "split-real-imag"
        assert opened.attrs["scpy_complex_real"] == "spectra__real"
        assert opened.attrs["scpy_complex_imag"] == "spectra__imag"
        assert "spectra__real" in opened.data_vars
        assert "spectra__imag" in opened.data_vars
        assert "spectra" not in opened.data_vars

    assert rebuilt.data.dtype == np.complex128
    assert np.array_equal(rebuilt.data, ds.data)


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_file_is_readable_by_xarray_open_dataset(tmp_path):
    ds = _make_dataset()
    filename = tmp_path / "dataset.nc"

    ds.to_netcdf(filename)

    with xr.open_dataset(filename, engine="scipy") as opened:
        assert opened.attrs["scpy_primary_variable"] == "spectra"
        assert opened.attrs["scpy_mask_variable"] == "spectra__mask"
        assert opened["spectra"].dims == ("y", "x")
        assert opened["spectra__mask"].dims == ("y", "x")


def test_netcdf_methods_raise_clear_error_when_xarray_is_missing(monkeypatch, tmp_path):
    missing = ImportError(
        "Missing optional dependency 'xarray'. Use conda or pip to install xarray."
    )
    original_import_optional_dependency = ndmodule.import_optional_dependency

    def fail_import(name, *args, **kwargs):
        if name == "xarray":
            raise missing
        return original_import_optional_dependency(name, *args, **kwargs)

    monkeypatch.setattr(ndmodule, "import_optional_dependency", fail_import)

    ds = _make_dataset()
    with pytest.raises(ImportError, match="Missing optional dependency 'xarray'"):
        ds.to_netcdf(tmp_path / "dataset.nc")

    with pytest.raises(ImportError, match="Missing optional dependency 'xarray'"):
        NDDataset.from_netcdf(tmp_path / "dataset.nc")
