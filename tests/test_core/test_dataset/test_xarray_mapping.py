# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import importlib.util

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
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
