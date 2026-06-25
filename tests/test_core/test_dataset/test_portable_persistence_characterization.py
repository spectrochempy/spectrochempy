"""Characterization tests for current portable xarray/NetCDF persistence."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset

if importlib.util.find_spec("xarray") is not None:
    import xarray as xr
else:  # pragma: no cover - depends on optional dependency availability
    xr = None

pytestmark = pytest.mark.skipif(xr is None, reason="xarray is not installed")


def _portable_attr_datetime(value):
    if value is None:
        return None
    return value.isoformat(sep=" ", timespec="seconds")


def _make_portable_metadata_dataset():
    # Kept local instead of reusing the broader semantic dataset helpers because
    # these characterization tests need a compact portable-specific fixture with
    # explicit provenance fields, acquisition_date, mask state, and nested
    # reader/vendor Meta payloads for xarray and NetCDF round-trips.
    ds = NDDataset(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=["y", "x"],
        coordset=[
            Coord([10.0, 20.0], name="y", title="time", units="s"),
            Coord([1000.0, 1200.0], name="x", title="wavenumber", units="cm^-1"),
        ],
        units="volt",
        mask=np.array([[False, True], [False, False]]),
        title="IR spectra",
        name="portable_demo",
        description="portable description",
        author="portable author",
        origin="portable origin",
        filename=Path("original_source.spc"),
        meta={
            "sample": "demo",
            "nested": {"temperature": 298.15, "labels": ["a", "b"]},
            "reader_metadata": {"reader": "omnic", "scan_count": 4},
            "vendor_metadata": {"firmware": "1.2.3", "serial": "abc"},
        },
        history="imported from vendor file",
    )
    ds.acquisition_date = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    ds._created = datetime(2024, 1, 2, 6, 7, 8, tzinfo=UTC)
    ds._modified = datetime(2024, 1, 2, 9, 10, 11, tzinfo=UTC)
    return ds


def _make_same_dim_dataset():
    # Kept local because same-dimension auxiliary-coordinate naming is itself
    # part of the portable characterization target and is clearer when the
    # fixture is declared inline next to the assertions that depend on it.
    coord_y = Coord([10.0, 20.0, 30.0], name="y", title="time", units="s")
    coord_x = Coord(
        [1000.0, 1100.0, 1200.0, 1300.0],
        name="x",
        title="wavenumber",
        units="cm^-1",
    )
    coord_x2 = Coord(
        [1.0, 1.1, 1.2, 1.3],
        name="x2",
        title="wavelength",
        units="um",
    )
    inner_x = CoordSet(coord_x, coord_x2, sorted=False)
    return NDDataset(
        np.arange(12.0).reshape(3, 4),
        dims=["y", "x"],
        coordset=[coord_y, inner_x],
        name="same_dim_demo",
    )


def test_to_xarray_currently_exports_aligned_identity_and_selected_provenance():
    ds = _make_portable_metadata_dataset()

    xds = ds.to_xarray()

    assert xds.attrs["scpy_name"] == ds.name
    assert xds.attrs["scpy_title"] == ds.title
    assert xds.attrs["scpy_description"] == ds.description
    assert xds.attrs["scpy_author"] == ds.author
    assert xds.attrs["scpy_origin"] == ds.origin
    assert xds.attrs["scpy_created"] == _portable_attr_datetime(ds._created)
    assert xds.attrs["scpy_modified"] == _portable_attr_datetime(ds._modified)
    assert xds.attrs["scpy_acquisition_date"] == _portable_attr_datetime(
        ds._acquisition_date
    )
    assert xds.attrs["scpy_meta"] == {
        "nested": {"labels": ["a", "b"], "temperature": 298.15},
        "reader_metadata": {"reader": "omnic", "scan_count": 4},
        "sample": "demo",
        "vendor_metadata": {"firmware": "1.2.3", "serial": "abc"},
    }
    assert "scpy_filename" not in xds.attrs
    assert xds.attrs["scpy_history"] == ds.history


def test_from_xarray_currently_preserves_aligned_provenance_and_only_still_loses_filename():
    ds = _make_portable_metadata_dataset()

    rebuilt = NDDataset.from_xarray(ds.to_xarray())

    assert tuple(rebuilt.dims) == tuple(ds.dims)
    assert np.array_equal(rebuilt.data, ds.data)
    assert np.array_equal(rebuilt.mask, ds.mask)
    assert rebuilt.name == ds.name
    assert rebuilt.title == ds.title
    assert rebuilt.description == ds.description
    assert rebuilt.origin == ds.origin
    assert rebuilt.author == ds.author
    assert rebuilt.created == ds.created
    assert rebuilt.modified == ds.modified
    assert rebuilt.acquisition_date == ds.acquisition_date
    assert rebuilt.filename == Path("portable_demo.scp")
    assert rebuilt.history == ds.history
    assert rebuilt.meta["nested"] == ds.meta["nested"]
    assert rebuilt.meta["reader_metadata"] == ds.meta["reader_metadata"]
    assert rebuilt.meta["vendor_metadata"] == ds.meta["vendor_metadata"]


def test_from_xarray_currently_transforms_same_dim_auxiliary_coordinate_names():
    ds = _make_same_dim_dataset()

    rebuilt = NDDataset.from_xarray(ds.to_xarray())

    rebuilt_inner = rebuilt.coord("x")
    rebuilt_names = [coord.name for coord in rebuilt_inner.coords]
    rebuilt_titles = [coord.title for coord in rebuilt_inner.coords]

    assert rebuilt_inner.is_same_dim
    assert rebuilt_titles == ["wavenumber", "wavelength"]
    assert "x2" not in rebuilt_names


def test_to_xarray_currently_normalizes_json_like_tuple_meta_to_lists():
    ds = NDDataset(
        np.array([1.0, 2.0, 3.0]),
        dims=["x"],
        coordset=[Coord([10.0, 20.0, 30.0], name="x", title="axis", units="s")],
        meta={
            "tuple_payload": ("a", 2),
            "numpy_int": np.int64(7),
            "reader_metadata": {"flag": True},
        },
    )

    rebuilt = NDDataset.from_xarray(ds.to_xarray())

    assert rebuilt.meta["tuple_payload"] == ["a", 2]
    assert rebuilt.meta["numpy_int"] == 7
    assert rebuilt.meta["reader_metadata"] == {"flag": True}


def test_to_xarray_currently_rejects_label_only_coordinates():
    ds = NDDataset(
        np.array([1.0, 2.0, 3.0]),
        dims=["x"],
        coordset=[Coord(labels=["A", "B", "C"], name="x", title="labels")],
        name="label_only_demo",
    )

    with pytest.raises(
        ValueError, match="different number of dimensions on data and dims"
    ):
        ds.to_xarray()


def test_from_xarray_currently_preserves_nullable_text_labels():
    ds = NDDataset(
        np.array([1.0, 2.0, 3.0]),
        dims=["x"],
        coordset=[Coord([10.0, 20.0, 30.0], name="x", title="axis", units="s")],
        name="labels_demo",
    )
    ds.coord("x").labels = ["A", None, "C"]

    xds = ds.to_xarray()
    rebuilt = NDDataset.from_xarray(xds)

    assert list(xds.coords["x_labels"].values) == ["A", "", "C"]
    assert list(rebuilt.coord("x").labels) == ["A", None, "C"]


def test_to_netcdf_and_from_netcdf_currently_preserve_aligned_provenance_and_only_still_lose_filename(
    tmp_path,
):
    ds = _make_portable_metadata_dataset()
    filename = tmp_path / "portable_demo.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    assert tuple(rebuilt.dims) == tuple(ds.dims)
    assert np.array_equal(rebuilt.data, ds.data)
    assert np.array_equal(rebuilt.mask, ds.mask)
    assert rebuilt.name == ds.name
    assert rebuilt.title == ds.title
    assert rebuilt.description == ds.description
    assert rebuilt.origin == ds.origin
    assert rebuilt.author == ds.author
    assert rebuilt.created == ds.created
    assert rebuilt.modified == ds.modified
    assert rebuilt.acquisition_date == ds.acquisition_date
    assert rebuilt.filename == Path("portable_demo.scp")
    assert rebuilt.history == ds.history
    assert rebuilt.meta["nested"] == ds.meta["nested"]
    assert rebuilt.meta["reader_metadata"] == ds.meta["reader_metadata"]
    assert rebuilt.meta["vendor_metadata"] == ds.meta["vendor_metadata"]


def test_netcdf_currently_omits_only_remaining_unaligned_filename_attr(tmp_path):
    ds = _make_portable_metadata_dataset()
    filename = tmp_path / "portable_demo.nc"

    ds.to_netcdf(filename)

    with xr.open_dataset(filename, engine="scipy") as opened:
        assert opened.attrs["scpy_name"] == ds.name
        assert opened.attrs["scpy_title"] == ds.title
        assert opened.attrs["scpy_description"] == ds.description
        assert opened.attrs["scpy_author"] == ds.author
        assert opened.attrs["scpy_origin"] == ds.origin
        assert opened.attrs["scpy_created"] == _portable_attr_datetime(ds._created)
        assert opened.attrs["scpy_modified"] == _portable_attr_datetime(ds._modified)
        assert opened.attrs["scpy_acquisition_date"] == _portable_attr_datetime(
            ds._acquisition_date
        )
        assert opened.attrs["scpy_history"] == json.dumps(ds.history, sort_keys=True)
        assert "scpy_filename" not in opened.attrs


def test_from_netcdf_currently_transforms_same_dim_auxiliary_coordinate_names(tmp_path):
    ds = _make_same_dim_dataset()
    filename = tmp_path / "same_dim_demo.nc"

    ds.to_netcdf(filename)
    rebuilt = NDDataset.from_netcdf(filename)

    rebuilt_inner = rebuilt.coord("x")
    rebuilt_names = [coord.name for coord in rebuilt_inner.coords]
    rebuilt_titles = [coord.title for coord in rebuilt_inner.coords]

    assert rebuilt_inner.is_same_dim
    assert rebuilt_titles == ["wavenumber", "wavelength"]
    assert "x2" not in rebuilt_names
