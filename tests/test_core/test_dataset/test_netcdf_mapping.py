# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import importlib.util
import json
from datetime import UTC
from datetime import datetime

import numpy as np
import pytest

import spectrochempy.core.dataset.nddataset as ndmodule
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset

if importlib.util.find_spec("xarray") is not None:
    import xarray as xr
else:  # pragma: no cover - depends on optional dependency availability
    xr = None


def _portable_attr_datetime(value):
    if value is None:
        return None
    return value.isoformat(sep=" ", timespec="seconds")


def _make_history_entries():
    return [
        (datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC), "Imported from instrument"),
        (datetime(2024, 1, 1, 8, 30, 0, tzinfo=UTC), "Baseline corrected"),
    ]


def _make_dataset(data=None, *, complex_data=False, acquisition_date=None):
    if data is None:
        if complex_data:
            data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
        else:
            data = np.array([[1.0, 2.0], [3.0, 4.0]])

    ds = NDDataset(
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
        description="portable description",
        author="portable author",
        origin="portable origin",
        meta={
            "sample": "demo",
            "nested": {"temperature": 298.15, "labels": ["a", "b"]},
            "count": 2,
        },
    )
    if acquisition_date is not None:
        ds.acquisition_date = acquisition_date
    ds._created = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    ds._modified = datetime(2024, 1, 2, 6, 7, 8, tzinfo=UTC)
    return ds


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_simple_float_dataset(tmp_path):
    ds = _make_dataset()
    filename = tmp_path / "dataset.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    assert np.array_equal(rebuilt.data, ds.data)
    assert tuple(rebuilt.dims) == tuple(ds.dims)


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_coordinates_identity_provenance_and_metadata(
    tmp_path,
):
    ds = _make_dataset(acquisition_date=datetime(2024, 1, 3, 9, 10, 11, tzinfo=UTC))
    ds._history = _make_history_entries()
    ds._modified = datetime(2024, 1, 2, 6, 7, 8, tzinfo=UTC)
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
    assert rebuilt.description == ds.description
    assert rebuilt.author == ds.author
    assert rebuilt.origin == ds.origin
    assert rebuilt.created == ds.created
    assert rebuilt.modified == ds.modified
    assert rebuilt.acquisition_date == ds.acquisition_date
    assert rebuilt.history == ds.history


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_empty_history_without_emitting_attr(tmp_path):
    ds = _make_dataset()
    filename = tmp_path / "dataset_empty_history.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    with xr.open_dataset(filename, engine="scipy") as opened:
        assert "scpy_history" not in opened.attrs

    assert rebuilt.history == []


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_single_history_entry(tmp_path):
    ds = _make_dataset()
    ds._history = _make_history_entries()[:1]
    ds._modified = datetime(2024, 1, 2, 6, 7, 8, tzinfo=UTC)
    filename = tmp_path / "dataset_single_history.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    assert rebuilt.history == ds.history


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_multi_entry_history(tmp_path):
    ds = _make_dataset()
    ds._history = _make_history_entries()
    ds._modified = datetime(2024, 1, 2, 6, 7, 8, tzinfo=UTC)
    filename = tmp_path / "dataset_multi_history.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    assert rebuilt.history == ds.history


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
    ds = _make_dataset(acquisition_date=datetime(2024, 1, 3, 9, 10, 11, tzinfo=UTC))
    ds._history = _make_history_entries()
    ds._modified = datetime(2024, 1, 2, 6, 7, 8, tzinfo=UTC)
    filename = tmp_path / "dataset.nc"

    ds.to_netcdf(filename)

    with xr.open_dataset(filename, engine="scipy") as opened:
        assert opened.attrs["scpy_primary_variable"] == "spectra"
        assert opened.attrs["scpy_mask_variable"] == "spectra__mask"
        assert opened.attrs["scpy_description"] == ds.description
        assert opened.attrs["scpy_author"] == ds.author
        assert opened.attrs["scpy_origin"] == ds.origin
        assert opened.attrs["scpy_created"] == _portable_attr_datetime(ds._created)
        assert opened.attrs["scpy_modified"] == _portable_attr_datetime(ds._modified)
        assert opened.attrs["scpy_acquisition_date"] == _portable_attr_datetime(
            ds._acquisition_date
        )
        assert opened.attrs["scpy_history"] == json.dumps(ds.history, sort_keys=True)
        assert opened["spectra"].dims == ("y", "x")
        assert opened["spectra__mask"].dims == ("y", "x")


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_naive_acquisition_date(tmp_path):
    ds = _make_dataset(acquisition_date=datetime(2024, 1, 3, 9, 10, 11))
    filename = tmp_path / "dataset_naive_acq.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    with xr.open_dataset(filename, engine="scipy") as opened:
        assert opened.attrs["scpy_acquisition_date"] == "2024-01-03 09:10:11"

    assert rebuilt.acquisition_date == ds.acquisition_date


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_from_netcdf_missing_timestamp_attrs_keeps_existing_fallback_behavior(tmp_path):
    ds = _make_dataset(acquisition_date=datetime(2024, 1, 3, 9, 10, 11, tzinfo=UTC))
    filename = tmp_path / "dataset_without_timestamp_attrs.nc"

    xds = ds.to_xarray()
    xds.attrs.pop("scpy_created")
    xds.attrs.pop("scpy_modified")
    xds.attrs.pop("scpy_acquisition_date")
    ndmodule._prepare_xarray_dataset_for_netcdf(xds).to_netcdf(filename, engine="scipy")

    rebuilt = NDDataset.from_netcdf(filename)

    assert rebuilt.created != ds.created
    assert rebuilt.modified != ds.modified
    assert rebuilt.acquisition_date is None


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_from_netcdf_missing_history_attr_keeps_existing_fallback_behavior(tmp_path):
    ds = _make_dataset()
    ds._history = _make_history_entries()
    ds._modified = datetime(2024, 1, 2, 6, 7, 8, tzinfo=UTC)
    filename = tmp_path / "dataset_without_history_attr.nc"

    xds = ds.to_xarray()
    xds.attrs.pop("scpy_history")
    ndmodule._prepare_xarray_dataset_for_netcdf(xds).to_netcdf(filename, engine="scipy")

    rebuilt = NDDataset.from_netcdf(filename)

    assert rebuilt.history == []


def _make_same_dim_netcdf_dataset():
    coord_y = Coord([10.0, 20.0, 30.0], name="y", title="time", units="s")
    coord_x = Coord(
        [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        name="x",
        title="wavenumber",
        units="cm^-1",
    )
    coord_x2 = Coord(
        [1.0, 1.25, 1.5, 1.75, 2.0], name="x2", title="wavelength", units="µm"
    )
    inner_x = CoordSet(coord_x, coord_x2, sorted=False)
    return NDDataset(
        np.random.default_rng(42).random((3, 5)),
        dims=["y", "x"],
        coordset=[coord_y, inner_x],
        name="spectra",
    )


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_same_dim_structure(tmp_path):
    ds = _make_same_dim_netcdf_dataset()
    filename = tmp_path / "same_dim.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    inner = rebuilt.coord("x")
    assert inner.is_same_dim
    assert len(inner.coords) == 2


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_roundtrip_preserves_auxiliary_coord_values(tmp_path):
    ds = _make_same_dim_netcdf_dataset()
    filename = tmp_path / "same_dim.nc"
    orig_inner = ds.coord("x")
    orig_aux_data = [c.data for c in orig_inner.coords if c is not orig_inner.default][
        0
    ]

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    new_inner = rebuilt.coord("x")
    new_aux_data = [c.data for c in new_inner.coords if c is not new_inner.default][0]
    assert np.allclose(new_aux_data, orig_aux_data)


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_netcdf_same_dim_roundtrip_preserves_default_coordinate(tmp_path):
    ds = _make_same_dim_netcdf_dataset()
    filename = tmp_path / "same_dim.nc"
    orig_inner = ds.coord("x")
    orig_default = orig_inner.default

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    new_inner = rebuilt.coord("x")
    assert new_inner.default is not None
    assert new_inner.default.title == orig_default.title
    assert np.allclose(new_inner.default.data, orig_default.data)


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
