# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import importlib.util

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


def _make_dataset(data=None, *, complex_data=False, unsupported_meta=False):
    if data is None:
        if complex_data:
            data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
        else:
            data = np.array([[1.0, 2.0], [3.0, 4.0]])

    meta = {
        "sample": "demo",
        "nested": {"temperature": 298.15, "labels": ["a", "b"]},
        "count": 2,
    }
    if unsupported_meta:
        meta["unsupported"] = object()

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
        meta=meta,
    )


def test_to_xarray_returns_dataset():
    ds = _make_dataset()

    xds = ds.to_xarray()

    assert isinstance(xds, xr.Dataset)
    assert xds.attrs["scpy_primary_variable"] == "spectra"


def test_xarray_roundtrip_preserves_numerical_data_dims_and_coordinates():
    ds = _make_dataset()

    xds = ds.to_xarray()
    rebuilt = NDDataset.from_xarray(xds)

    assert np.array_equal(rebuilt.data, ds.data)
    assert tuple(rebuilt.dims) == tuple(ds.dims)
    assert np.array_equal(rebuilt.coord("x").data, ds.coord("x").data)
    assert np.array_equal(rebuilt.coord("y").data, ds.coord("y").data)


def test_xarray_roundtrip_preserves_units_mask_title_and_name():
    ds = _make_dataset()

    xds = ds.to_xarray()
    rebuilt = NDDataset.from_xarray(xds)

    assert str(rebuilt.units) == str(ds.units)
    assert np.array_equal(rebuilt.mask, ds.mask)
    assert rebuilt.title == ds.title
    assert rebuilt.name == ds.name
    assert str(rebuilt.coord("x").units) == str(ds.coord("x").units)
    assert rebuilt.coord("x").title == ds.coord("x").title


def test_xarray_roundtrip_preserves_json_compatible_metadata():
    ds = _make_dataset()

    xds = ds.to_xarray()
    rebuilt = NDDataset.from_xarray(xds)

    assert xds.attrs["scpy_meta"] == {
        "count": 2,
        "nested": {"labels": ["a", "b"], "temperature": 298.15},
        "sample": "demo",
    }
    assert rebuilt.meta["sample"] == "demo"
    assert rebuilt.meta["nested"]["temperature"] == 298.15
    assert rebuilt.meta["nested"]["labels"] == ["a", "b"]


def test_xarray_export_skips_unsupported_metadata():
    ds = _make_dataset(unsupported_meta=True)

    xds = ds.to_xarray()
    rebuilt = NDDataset.from_xarray(xds)

    assert xds.attrs["scpy_skipped_meta_keys"] == ["unsupported"]
    assert "unsupported" not in xds.attrs.get("scpy_meta", {})
    assert rebuilt.meta["unsupported"] is None
    assert rebuilt.meta["sample"] == "demo"


def test_xarray_roundtrip_preserves_complex_dtype():
    ds = _make_dataset(complex_data=True)

    xds = ds.to_xarray()
    rebuilt = NDDataset.from_xarray(xds)

    assert xds["spectra"].dtype == np.complex128
    assert rebuilt.data.dtype == np.complex128
    assert np.array_equal(rebuilt.data, ds.data)


def _make_same_dim_dataset():
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
class TestSameDimCoordRoundtrip:
    def test_export_contains_auxiliary_coord(self):
        ds = _make_same_dim_dataset()
        xds = ds.to_xarray()
        aux_names = [
            n
            for n in xds.coords
            if xds.coords[n].attrs.get("scpy_coord_role") == "auxiliary"
        ]
        assert len(aux_names) == 1
        assert aux_names[0] == "x_aux_0001"

    def test_export_auxiliary_coord_has_correct_attrs(self):
        ds = _make_same_dim_dataset()
        xds = ds.to_xarray()
        # Aux coord is _2 = wavenumber (cm⁻¹) — default is _1 = wavelength (µm)
        aux = xds.coords["x_aux_0001"]
        assert aux.attrs["scpy_coord_role"] == "auxiliary"
        assert aux.attrs["scpy_owner_dim"] == "x"
        assert aux.attrs["units"] == "cm⁻¹"
        assert aux.attrs["scpy_title"] == "wavenumber"

    def test_export_dim_coord_has_default_marker(self):
        ds = _make_same_dim_dataset()
        xds = ds.to_xarray()
        assert xds.coords["x"].attrs["scpy_coord_role"] == "default"
        assert xds.coords["x"].attrs["scpy_default"] == "x"

    def test_roundtrip_preserves_same_dim_structure(self):
        ds = _make_same_dim_dataset()
        xds = ds.to_xarray()
        rebuilt = NDDataset.from_xarray(xds)
        inner = rebuilt.coord("x")
        assert hasattr(inner, "is_same_dim")
        assert inner.is_same_dim
        assert len(inner.coords) == 2

    def test_roundtrip_preserves_default_coordinate(self):
        ds = _make_same_dim_dataset()
        orig_inner = ds.coord("x")
        orig_default = orig_inner.default
        xds = ds.to_xarray()
        rebuilt = NDDataset.from_xarray(xds)
        inner = rebuilt.coord("x")
        assert inner.default is not None
        assert inner.default.title == orig_default.title
        assert str(inner.default.units) == str(orig_default.units)
        assert np.allclose(inner.default.data, orig_default.data)

    def test_roundtrip_preserves_auxiliary_coordinate_values(self):
        ds = _make_same_dim_dataset()
        orig_inner = ds.coord("x")
        xds = ds.to_xarray()
        rebuilt = NDDataset.from_xarray(xds)
        new_inner = rebuilt.coord("x")
        # Find the aux coord (non-default) in rebuilt and verify values
        orig_aux_data = [
            c.data for c in orig_inner.coords if c is not orig_inner.default
        ][0]
        new_aux_data = [c.data for c in new_inner.coords if c is not new_inner.default][
            0
        ]
        assert np.allclose(new_aux_data, orig_aux_data)

    def test_roundtrip_preserves_numerical_data(self):
        ds = _make_same_dim_dataset()
        xds = ds.to_xarray()
        rebuilt = NDDataset.from_xarray(xds)
        assert np.allclose(rebuilt.data, ds.data)

    def test_roundtrip_preserves_aux_coord_ordering(self):
        coord_x3 = Coord([0.5, 1.0, 1.5, 2.0, 2.5], title="energy", units="eV")
        inner_x3 = CoordSet(
            Coord(
                [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                title="wavenumber",
                units="cm^-1",
            ),
            Coord([1.0, 1.25, 1.5, 1.75, 2.0], title="wavelength", units="µm"),
            coord_x3,
            sorted=False,
        )
        ds = NDDataset(
            np.random.default_rng(42).random((3, 5)),
            dims=["y", "x"],
            coordset=[
                Coord([10.0, 20.0, 30.0], title="time", units="s"),
                inner_x3,
            ],
        )
        orig_titles = {c.title for c in ds.coord("x").coords}
        xds = ds.to_xarray()
        rebuilt = NDDataset.from_xarray(xds)
        new_titles = {c.title for c in rebuilt.coord("x").coords}
        assert new_titles == orig_titles
        assert len(new_titles) == 3

    def test_roundtrip_multiple_dims_with_aux(self):
        coord_y = Coord([10.0, 20.0, 30.0], title="time", units="s")
        coord_y2 = Coord([0.0, 50.0, 100.0], title="temperature", units="C")
        inner_y = CoordSet(coord_y2, coord_y, sorted=False)
        coord_x = Coord(
            [1000.0, 1100.0, 1200.0, 1300.0, 1400.0], title="wavenumber", units="cm^-1"
        )
        coord_x2 = Coord([1.0, 1.25, 1.5, 1.75, 2.0], title="wavelength", units="µm")
        inner_x = CoordSet(coord_x, coord_x2, sorted=False)
        ds = NDDataset(
            np.random.default_rng(42).random((3, 5)),
            dims=["y", "x"],
            coordset=[inner_y, inner_x],
        )
        xds = ds.to_xarray()
        rebuilt = NDDataset.from_xarray(xds)
        for dim in ("y", "x"):
            orig_inner = ds.coord(dim)
            new_inner = rebuilt.coord(dim)
            assert new_inner.is_same_dim
            assert len(new_inner.coords) == len(orig_inner.coords)
        assert np.allclose(rebuilt.data, ds.data)

    def test_default_independent_of_construction_order(self):
        """
        Default after round-trip is always the xarray dim coord,
        regardless of the original CoordSet pass order.
        """
        coord_y = Coord([10.0, 20.0, 30.0], name="y", title="time", units="s")
        # Two coords with different names and data values
        wavenumber = Coord(
            [1000.0, 1100.0], name="x", title="wavenumber", units="cm^-1"
        )
        wavelength = Coord([1.0, 1.25], name="x2", title="wavelength", units="µm")
        pass_orders = [
            [wavenumber, wavelength],
            [wavelength, wavenumber],
        ]
        for coords in pass_orders:
            inner_x = CoordSet(*coords, sorted=False)
            ds = NDDataset(
                np.random.default_rng(42).random((3, 2)),
                dims=["y", "x"],
                coordset=[coord_y, inner_x],
            )
            # Determine which coord became the dim coord (default) in this original
            orig_default = ds.coord("x").default
            xds = ds.to_xarray()
            rebuilt = NDDataset.from_xarray(xds)
            new_default = rebuilt.coord("x").default
            assert new_default.title == orig_default.title
            assert np.allclose(new_default.data, orig_default.data)
