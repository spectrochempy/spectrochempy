# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Test ndcoordset
"""

from copy import copy

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import DimensionalityError, ur
from spectrochempy.utils.testing import assert_coord_almost_equal


# ======================================================================================
# CoordSet
# ======================================================================================
def test_coordset_init(coord0, coord1, coord2):
    coord3 = coord2.copy()
    coord3.title = "titi"

    coordsa = CoordSet(coord0, coord3, coord2)  # First syntax

    assert coordsa.names == ["x", "y", "z"]  # coordinates are sorted in the coordset

    coordsb = CoordSet(
        (coord0, coord3, coord2)
    )  # second syntax with a tuple of coordinates
    assert coordsb.names == ["x", "y", "z"]

    # but warning
    coordsa1 = CoordSet(
        [coord0[:3], coord3[:3], coord2[:3]]
    )  # A list means that it is a sub-coordset (different meaning)
    assert coordsa1.names == ["x"]
    assert coordsa1.x.names == ["_1", "_2", "_3"]

    coordsc = CoordSet(x=coord2, y=coord3, z=coord0)  # third syntax
    assert coordsc.names == ["x", "y", "z"]

    coordsc1 = CoordSet({"x": coord2, "y": coord3, "z": coord0})
    assert coordsc1.names == ["x", "y", "z"]

    coordsd = CoordSet(
        coord3, x=coord2, y=coord3, z=coord0
    )  # conflict (keyw replace args)
    assert coordsa == coordsb
    assert coordsa == coordsc
    assert coordsa == coordsd
    assert coordsa == coordsc1
    c = coordsa["x"]
    assert c == coord2
    c = coordsa["y"]
    assert c == coord3
    assert coordsa["wavenumber"] == coord0

    coord4 = copy(coord2)
    coordsc = CoordSet([coord1[:3], coord2[:3], coord4[:3]])
    assert coordsa != coordsc

    coordse = CoordSet(
        x=(coord1[:3], coord2[:3]), y=coord3, z=coord0
    )  # coordset as coordinates
    assert coordse["x"].titles == CoordSet(coord1, coord2).titles
    assert coordse["x_1"] == coord2
    assert coordse["titi"] == coord3

    # iteration
    for coord in coordsa:
        assert isinstance(coord, Coord)

    for i, coord in enumerate(coordsa):
        assert isinstance(coord, Coord)

    assert repr(coord0) in [
        "Coord: [float64] cm⁻¹ (size: 10)",
        "Coord: [float64] 1/cm (size: 10)",
    ]

    coords = CoordSet(coord0.copy(), coord0)

    assert repr(coords).startswith("CoordSet: [x:wavenumber, y:wavenumber]")

    with pytest.raises(ValueError):
        coords = CoordSet(2, 3)  # Coord in CoordSet cannot be simple scalar

    coords = CoordSet(x=coord2, y=coord3, z=None)
    assert coords.names == ["x", "y", "z"]
    assert coords.z.is_empty

    coords = CoordSet(x=coord2, y=coord3, z=np.array((1, 2, 3)))
    assert coords.names == ["x", "y", "z"]
    assert coords.z.size == 3

    with pytest.raises(KeyError):
        coords = CoordSet(
            x=coord2, y=coord3, fx=np.array((1, 2, 3))
        )  # wrong key (must be a single char)

    with pytest.raises(ValueError):
        coords = CoordSet(x=coord2, y=coord3, z=3)  # wrong coordinate value

    # set a coordset from another one
    coords = CoordSet(**coordse)
    assert coordse.names == ["x", "y", "z"]
    assert coords.names == ["x", "y", "z"]
    assert coords == coordse

    # not recommended
    coords2 = CoordSet(*coordse)  # loose the names so the ordering may be different
    assert coords2.names == ["x", "y", "z"]
    assert coords.x == coords2.z


def test_coordset_multicoord_for_a_single_dim():
    # normal coord (single numerical array for a axis)

    coord1 = Coord(
        data=np.linspace(1000.0, 4000.0, 5),
        labels="a b c d e".split(),
        mask=None,
        units="cm^1",
        title="wavelengths",
    )

    coord0 = Coord(
        data=np.linspace(20, 500, 5),
        labels="very low-low-normal-high-very high".split("-"),
        mask=None,
        units="K",
        title="temperature",
    )

    # pass as a list of coord -> this become a subcoordset
    coordsa = CoordSet([coord0, coord1])
    assert (
        repr(coordsa) == "CoordSet: [x:[_1:temperature, _2:wavelengths]]"
    )  # note the internal coordinates are not sorted
    assert not coordsa.is_same_dim
    assert coordsa.x.is_same_dim

    coordsb = coordsa.x

    # try to pass arguments, each being an coord
    coordsc = CoordSet(coord1, coord0)
    assert not coordsc.is_same_dim
    assert repr(coordsc) == "CoordSet: [x:temperature, y:wavelengths]"

    # try to pass arguments where each are a coords
    coordsd = CoordSet(coordsa.x, coordsc)
    assert (
        repr(coordsd)
        == "CoordSet: [x:[_1:temperature, _2:wavelengths], y:[_1:temperature, _2:wavelengths]]"
    )

    assert not coordsd.is_same_dim
    assert np.all([item.is_same_dim for item in coordsd])

    coordse = CoordSet(coordsb, coord1)
    assert (
        repr(coordse) == "CoordSet: [x:wavelengths, y:[_1:temperature, _2:wavelengths]]"
    )

    assert not coordse.is_same_dim
    assert coordse("y").is_same_dim

    co = coordse("x")
    assert isinstance(co, Coord)

    co = coordse("y")
    assert isinstance(co, CoordSet)
    assert co.name == "y"
    assert co.names == ["_1", "_2"]
    assert co._1 == coord0

    co = coordse[-1]
    assert isinstance(co, CoordSet)
    assert co.name == "y"  # should keep the original name (solved)
    assert co["_1"] == coord0


def test_coordset_call(coord0, coord1):
    coordsa = CoordSet(coord0, coord1)

    assert str(coordsa) == "CoordSet: [x:time-on-stream, y:wavenumber]"
    a = coordsa(1, 0)
    assert a == coordsa

    b = coordsa(1)
    assert b == coord0  # reordering

    c = coordsa("x")
    assert c == coord1

    d = coordsa("time-on-stream")
    assert d == coord1

    with pytest.raises(KeyError):
        coordsa("x_a")  # do not exit

    coordsa("y_a")


def test_coordset_get(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)

    coord = coords["temperature"]
    assert str(coord) == "Coord: [float64] K (size: 3)"
    assert coord.name == "z"

    coord = coords["wavenumber"]
    assert coord.name == "_1"

    coord = coords["y_2"]
    assert coord.name == "_2"

    coord = coords["_1"]
    assert coord.name == "_1"


def test_coordset_del(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)

    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:wavenumber], z:temperature]"
    )

    del coords["temperature"]
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:wavenumber]]"
    )

    del coords.y["wavenumber"]
    assert (
        str(coords) == repr(coords) == "CoordSet: [x:time-on-stream, y:[_2:wavenumber]]"
    )

    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del coords["wavenumber"]
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream, y:[_2:wavenumber], z:temperature]"
    )

    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del coords.y_2
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber], z:temperature]"
    )

    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del coords.y._1
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream, y:[_2:wavenumber], z:temperature]"
    )


def test_coordset_copy(coord0, coord1):
    coord2 = Coord.linspace(200.0, 300.0, 3, units="K", title="temperature")

    coordsa = CoordSet(coord0, coord1, coord2)

    coordsb = coordsa.copy()
    assert coordsa == coordsb
    assert coordsa is not coordsb
    assert coordsa(1) == coordsb(1)
    assert coordsa(1).name == coordsb(1).name

    # copy
    coords = CoordSet(coord0, coord0.copy())
    coords1 = coords[:]
    assert coords is not coords1

    import copy

    coords2 = copy.deepcopy(coords)
    assert coords == coords2


def test_coordset_implements(coord0, coord1):
    coordsa = CoordSet(coord0, coord1)

    assert coordsa._implements("CoordSet")
    assert coordsa._implements() == "CoordSet"


def test_coordset_sizes(coord0, coord1):
    coords = CoordSet(coord0, coord1)
    assert coords.sizes == [coords.x.size, coords.y.size] == [coord1.size, coord0.size]

    coords = CoordSet([coord0, coord0.copy()], coord1)
    assert coords.sizes == [coords.x.size, coords.y.size] == [coord1.size, coord0.size]

    assert coord0.size != coord0[:7].size
    with pytest.raises(ValueError):
        coords = CoordSet([coord0, coord0[:7]], coord1)


def test_coordset_update(coord0, coord1):
    coords = CoordSet(coord0, coord1)

    coords.update(x=coord0)

    assert coords[1] == coords[0] == coord0


def test_coordset_str_repr(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)

    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:wavenumber], z:temperature]"
    )
    assert repr(coords) == str(coords)


def test_coordset_set(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:wavenumber], z:temperature]"
    )

    coords.set_titles("time", "dddd", "celsius")
    assert (
        str(coords) == "CoordSet: [x:time, y:[_1:wavenumber, _2:wavenumber], z:celsius]"
    )

    coords.set_titles(x="time", z="celsius", y_1="length")
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time, y:[_1:length, _2:wavenumber], z:celsius]"
    )

    coords.set_titles("t", ("l", "g"), x="x")
    assert str(coords) == "CoordSet: [x:x, y:[_1:l, _2:g], z:celsius]"

    coords.set_titles(("t", ("l", "g")), z="z")
    assert str(coords) == "CoordSet: [x:t, y:[_1:l, _2:g], z:z]"

    coords.set_titles()  # nothing happens
    assert str(coords) == "CoordSet: [x:t, y:[_1:l, _2:g], z:z]"

    with pytest.raises(DimensionalityError):  # because units doesn't match
        coords.set_units(("km/s", ("s", "m")), z="radian")

    coords.set_units(("km/s", ("s", "m")), z="radian", force=True)  # force change
    assert str(coords) == "CoordSet: [x:t, y:[_1:l, _2:wavelength], z:z]"
    assert coords.y_1.units == ur("s")

    # set item

    coords["z"] = coord2
    assert str(coords) == "CoordSet: [x:t, y:[_1:l, _2:wavelength], z:temperature]"

    coords["temperature"] = coord1
    assert str(coords) == "CoordSet: [x:t, y:[_1:l, _2:wavelength], z:time-on-stream]"

    coords["y_2"] = coord2
    assert str(coords) == "CoordSet: [x:t, y:[_1:l, _2:temperature], z:time-on-stream]"

    coords["_1"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:t, y:[_1:temperature, _2:temperature], z:time-on-stream]"
    )

    coords["t"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:temperature, y:[_1:temperature, _2:temperature], z:time-on-stream]"
    )

    coord2.title = "zaza"
    coords["temperature"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:zaza, y:[_1:temperature, _2:temperature], z:time-on-stream]"
    )

    coords["temperature"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:zaza, y:[_1:zaza, _2:temperature], z:time-on-stream]"
    )

    coords.set(coord1, coord0, coord2)
    assert str(coords) == "CoordSet: [x:zaza, y:wavenumber, z:time-on-stream]"

    coords.z = coord0
    assert str(coords) == "CoordSet: [x:zaza, y:wavenumber, z:wavenumber]"

    coords.zaza = coord0
    assert str(coords) == "CoordSet: [x:wavenumber, y:wavenumber, z:wavenumber]"

    coords.wavenumber = coord2
    assert str(coords) == "CoordSet: [x:zaza, y:wavenumber, z:wavenumber]"


def test_issue_310():
    import numpy as np

    import spectrochempy as scp

    D = scp.NDDataset(np.zeros((10, 5)))
    c5 = scp.Coord.linspace(start=0.0, stop=1000.0, num=5, name="xcoord")
    c10 = scp.Coord.linspace(start=0.0, stop=1000.0, num=10, name="ycoord")

    assert c5.name == "xcoord"
    D.set_coordset(x=c5, y=c10)
    assert c5.name == "x"
    assert str(D.coordset) == "CoordSet: [x:<untitled>, y:<untitled>]"
    assert D.dims == ["y", "x"]

    E = scp.NDDataset(np.ndarray((10, 5)))
    E.dims = ["t", "s"]
    E.set_coordset(s=c5, t=c10)
    assert c5.name == "s"
    assert str(E.coordset) == "CoordSet: [s:<untitled>, t:<untitled>]"
    assert E.dims == ["t", "s"]

    E = scp.NDDataset(np.ndarray((5, 10)))
    E.dims = ["s", "t"]
    E.set_coordset(s=c5, t=c10)
    assert c5.name == "s"
    assert str(E.coordset) == "CoordSet: [s:<untitled>, t:<untitled>]"
    assert E.dims == ["s", "t"]

    assert str(D.coordset) == "CoordSet: [x:<untitled>, y:<untitled>]"
    assert D.dims == ["y", "x"]

    assert str(D[:, 1]) == "NDDataset: [float64] unitless (shape: (y:10, x:1))"


def test_coordset_arithmetics():
    # typical use case
    ds = NDDataset([0.0, 1.0, 2.0])
    x2 = Coord(np.array([0.5, 0.8, 9.0]))
    x1 = Coord(np.array([1.5, 5.8, -9.0]))
    ds.x = CoordSet(Coord(x2), Coord(x1))

    diff = ds.x - ds.x[0]

    assert_coord_almost_equal(diff["_1"], x1 - x1[0])
    assert_coord_almost_equal(diff["_2"], x2 - x2[0])

    # in the following, the coordinates are not considered by default
    # as multi coordinates, so the slicing returns a Coord, not a
    # slices Coorset:
    x = CoordSet(Coord(x2), Coord(x1))
    try:
        diff = x - x[0]
    except Exception as e:
        assert type(e) is NotImplementedError

    # but we can declare the coordinates as multi-coordinates:
    x._is_same_dim = True
    diff = x - x[0]
    assert_coord_almost_equal(diff["_1"], x1 - x1[0])
    assert_coord_almost_equal(diff["_2"], x2 - x2[0])
