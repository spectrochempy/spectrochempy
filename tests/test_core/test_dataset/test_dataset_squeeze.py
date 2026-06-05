# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.testing import assert_array_equal


def test_squeeze_all_dims_have_coords():
    """Squeeze preserves coords when every dim has an explicit coord."""
    coord_y = scp.Coord(np.array([1.0]), title="y_dim")
    coord_x = scp.Coord(np.array([100.0, 200.0, 300.0, 400.0]), title="wavelength")
    ds = scp.NDDataset(np.ones((1, 4)), coordset=[coord_y, coord_x])
    assert ds.shape == (1, 4)
    assert ds.dims == ["y", "x"]
    squeezed = ds.squeeze()
    assert squeezed.shape == (4,)
    assert squeezed.dims == ["x"]
    assert_array_equal(squeezed.x.data, np.array([100.0, 200.0, 300.0, 400.0]))
    assert squeezed.x.title == "wavelength"


def test_squeeze_partial_coords_no_name_mismatch():
    """Squeeze handles singleton dims that have no explicit coord."""
    coord_x = scp.Coord(np.array([100.0, 200.0, 300.0, 400.0]), title="wavelength")
    ds = scp.NDDataset(np.ones((1, 4)), coordset=[coord_x])
    squeezed = ds.squeeze()
    assert squeezed.shape == (4,)
    assert squeezed.dims == ["x"]
    assert_array_equal(squeezed.x.data, np.array([100.0, 200.0, 300.0, 400.0]))
    assert squeezed.x.title == "wavelength"


def test_squeeze_multiple_singletons():
    """Squeeze removes all singleton dims, preserving only non-singleton coords."""
    coord = scp.Coord(np.array([100.0, 200.0, 300.0, 400.0]), title="wavelength")
    # dims for 4D: ['u', 'z', 'y', 'x']; coord at index 2 -> dim 'y'
    ds = scp.NDDataset(np.ones((1, 1, 4, 1)), coordset=[None, None, coord, None])
    squeezed = ds.squeeze()
    assert squeezed.shape == (4,)
    assert squeezed.dims == ["y"]
    assert_array_equal(squeezed.y.data, np.array([100.0, 200.0, 300.0, 400.0]))
    assert squeezed.y.title == "wavelength"


def test_squeeze_named_dims_coord_consistency():
    """Squeeze in 3D with partial coords: coord-to-dim mapping stays correct."""
    coord_z = scp.Coord(np.array([10.0, 20.0, 30.0]), title="time")
    coord_x = scp.Coord(np.array([1.0, 2.0, 3.0]), title="temperature")
    ds = scp.NDDataset(np.ones((3, 1, 3)), coordset=[coord_z, None, coord_x])
    assert ds.shape == (3, 1, 3)
    assert ds.dims == ["z", "y", "x"]
    squeezed = ds.squeeze()
    assert squeezed.shape == (3, 3)
    assert squeezed.dims == ["z", "x"]
    assert_array_equal(squeezed.z.data, np.array([10.0, 20.0, 30.0]))
    assert_array_equal(squeezed.x.data, np.array([1.0, 2.0, 3.0]))
    assert squeezed.z.title == "time"
    assert squeezed.x.title == "temperature"


def test_squeeze_preserves_remaining_multicoord_dimension():
    """Squeeze keeps same-dimension multi-coordinates on surviving dims."""
    coord_y = scp.Coord(np.array([0.0]), title="row")
    coord_x_primary = scp.Coord(np.array([10.0, 20.0, 30.0]), title="wavelength")
    coord_x_secondary = scp.Coord(np.array([1.0, 2.0, 3.0]), title="index")
    ds = scp.NDDataset(np.ones((1, 3)), coordset=[coord_y, coord_x_primary])
    ds.x = scp.CoordSet(coord_x_secondary.copy(), coord_x_primary.copy())
    ds.x.select(2)

    squeezed = ds.squeeze()

    assert squeezed.shape == (3,)
    assert squeezed.dims == ["x"]
    assert isinstance(squeezed.x, scp.CoordSet)
    assert squeezed.x.is_same_dim
    assert squeezed.x.default == squeezed.x._2
    assert_array_equal(squeezed.x._1.data, np.array([10.0, 20.0, 30.0]))
    assert_array_equal(squeezed.x._2.data, np.array([1.0, 2.0, 3.0]))
    assert squeezed.x._1.title == "wavelength"
    assert squeezed.x._2.title == "index"


if __name__ == "__main__":
    pytest.main([__file__])
