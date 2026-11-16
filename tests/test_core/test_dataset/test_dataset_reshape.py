# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.testing import assert_array_equal


def test_reshape_noop_preserves_coordinates():
    coord_y = scp.Coord(np.array([1.0, 2.0, 3.0]), title="time")
    coord_x = scp.Coord(np.array([10.0, 20.0]), title="wavelength")
    ds = scp.NDDataset(np.arange(6.0).reshape(3, 2), coordset=[coord_y, coord_x])

    reshaped = ds.reshape((3, 2))

    assert reshaped.shape == (3, 2)
    assert reshaped.dims == ["y", "x"]
    assert_array_equal(reshaped.y.data, coord_y.data)
    assert_array_equal(reshaped.x.data, coord_x.data)
    assert reshaped.y.title == "time"
    assert reshaped.x.title == "wavelength"


def test_reshape_drop_policy_clears_coordset():
    coord_y = scp.Coord(np.array([1.0, 2.0, 3.0]), title="time")
    coord_x = scp.Coord(np.array([10.0, 20.0]), title="wavelength")
    ds = scp.NDDataset(np.arange(6.0).reshape(3, 2), coordset=[coord_y, coord_x])

    reshaped = ds.reshape((6,), coord_policy="drop")

    assert reshaped.shape == (6,)
    assert reshaped.coordset is None


def test_reshape_with_explicit_dims_and_coords_override():
    coord_y = scp.Coord(np.array([1.0, 2.0, 3.0]), title="time")
    coord_x = scp.Coord(np.array([10.0, 20.0]), title="wavelength")
    ds = scp.NDDataset(np.arange(6.0).reshape(3, 2), coordset=[coord_y, coord_x])
    cycle = scp.Coord(np.array([1.0]), title="cycle index")

    reshaped = ds.reshape(
        (1, 3, 2),
        dims=("cycle", "y", "x"),
        coords={"cycle": cycle},
    )

    assert reshaped.shape == (1, 3, 2)
    assert reshaped.dims == ["cycle", "y", "x"]
    assert_array_equal(reshaped.coordset["cycle"].data, [1.0])
    assert reshaped.coordset["cycle"].title == "cycle index"
    assert_array_equal(reshaped.y.data, coord_y.data)
    assert_array_equal(reshaped.x.data, coord_x.data)


def test_reshape_without_source_coordset_keeps_coordset_none():
    ds = scp.NDDataset(np.arange(6.0).reshape(3, 2))

    reshaped = ds.reshape((1, 3, 2), dims=("u", "y", "x"))

    assert reshaped.shape == (1, 3, 2)
    assert reshaped.coordset is None


def test_reshape_strict_preserves_current_ambiguity_error():
    coord_y = scp.Coord(np.array([1.0, 2.0]), title="rows")
    coord_x = scp.Coord(np.array([10.0, 20.0]), title="cols")
    ds = scp.NDDataset(np.arange(4.0).reshape(2, 2), coordset=[coord_y, coord_x])

    with pytest.raises(
        ValueError,
        match="strict mode: cannot unambiguously map dim",
    ):
        ds.reshape((2, 2), coord_policy="strict")


def test_reshape_strict_preserves_unambiguous_coordinates():
    coord_y = scp.Coord(np.array([1.0, 2.0, 3.0]), title="rows")
    coord_x = scp.Coord(np.array([10.0, 20.0]), title="cols")
    ds = scp.NDDataset(np.arange(6.0).reshape(3, 2), coordset=[coord_y, coord_x])

    reshaped = ds.reshape((3, 2), coord_policy="strict")

    assert reshaped.shape == (3, 2)
    assert_array_equal(reshaped.y.data, coord_y.data)
    assert_array_equal(reshaped.x.data, coord_x.data)


def test_reshape_preserves_same_dimension_multicoord_when_dim_survives():
    coord_y = scp.Coord(np.array([1.0, 2.0, 3.0]), title="time")
    coord_x_primary = scp.Coord(np.array([10.0, 20.0]), title="wavelength")
    coord_x_secondary = scp.Coord(np.array([100.0, 200.0]), title="index")
    ds = scp.NDDataset(
        np.arange(6.0).reshape(3, 2), coordset=[coord_y, coord_x_primary]
    )
    ds.x = scp.CoordSet(coord_x_secondary.copy(), coord_x_primary.copy())
    ds.x.select(2)

    reshaped = ds.reshape((1, 3, 2), dims=("u", "y", "x"))

    assert reshaped.shape == (1, 3, 2)
    assert isinstance(reshaped.x, scp.CoordSet)
    assert reshaped.x.is_same_dim
    assert reshaped.x.names == ["_1", "_2"]
    assert reshaped.x.default == reshaped.x._2
    assert_array_equal(reshaped.x._1.data, coord_x_primary.data)
    assert_array_equal(reshaped.x._2.data, coord_x_secondary.data)
    assert reshaped.x._1.title == coord_x_primary.title
    assert reshaped.x._2.title == coord_x_secondary.title
    assert reshaped.x._1._parent_dim == "x"
    assert reshaped.x._2._parent_dim == "x"


def test_reshape_preserves_current_reference_removal_behavior():
    coord_x = scp.Coord(np.array([1.0, 2.0, 3.0]), title="time")
    ds = scp.NDDataset(
        np.arange(3.0),
        coordset=scp.CoordSet(x=coord_x, y="x"),
    )

    reshaped = ds.reshape((3,))

    assert reshaped.shape == (3,)
    assert reshaped.coordset.references == {}
    assert_array_equal(reshaped.x.data, coord_x.data)


def test_split_reshape_keeps_block_geometry_coordinates_and_mask():
    data = scp.NDDataset(np.arange(30.0).reshape(10, 3))
    data.y = [
        scp.Coord(np.arange(10) * 2 + 3, title="time", units="ms"),
        scp.Coord(np.arange(10) * 3, title="elapsed", units="s"),
    ]
    data.y.select(2)
    data.x = scp.Coord([10, 20, 30], units="eV")
    data[8, 2] = scp.MASKED

    grouped = data.reshape((-1, 5, 3), dims=("cycle", "y", "x"), coord_policy="split")

    assert grouped.shape == (2, 5, 3)
    assert grouped.mask[1, 3, 2]
    assert grouped.x == data.x
    old_default = data.y.default
    assert grouped.y.default.units == old_default.units
    assert_array_equal(grouped.coordset["cycle"].default.data, old_default.data[::5])
    assert_array_equal(
        grouped.y.default.data, old_default.data[:5] - old_default.data[0]
    )


def test_split_reshape_requires_an_unambiguous_axis_and_numeric_coordinate():
    data = scp.NDDataset(np.ones((6, 4)))
    with pytest.raises(ValueError, match="numeric coordinate"):
        data.reshape((2, 3, 4), coord_policy="split")

    data.y = scp.Coord(np.arange(6))
    with pytest.raises(ValueError, match="exactly one axis"):
        data.reshape((2, 2, 6), coord_policy="split")
