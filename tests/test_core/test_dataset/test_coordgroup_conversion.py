# SPDX-License-Identifier: BSD-3-Clause
# (see LICENSE.txt for details)

"""Focused tests for private CoordSet group-model conversion helpers."""

from spectrochempy.core.dataset._coordgroup import _coordset_to_groups
from spectrochempy.core.dataset._coordgroup import _groups_to_coordset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.utils.testing import assert_array_equal


def _assert_public_coord_equal(left, right):
    assert left.name == right.name
    assert left.title == right.title
    assert left.units == right.units
    assert_array_equal(left.data, right.data)
    if left.labels is None:
        assert right.labels is None
    else:
        assert_array_equal(left.labels, right.labels)


def _assert_public_coordset_equal(left, right):
    assert left.names == right.names
    assert left.keys() == right.keys()
    assert left.titles == right.titles
    assert left.units == right.units
    assert_array_equal(left.data, right.data)


def test_simple_coordset_group_roundtrip_preserves_public_behavior():
    coord = Coord(
        [1.0, 2.0, 3.0],
        name="x",
        title="distance",
        units="cm",
        labels=["a", "b", "c"],
    )
    coordset = CoordSet(x=coord)

    groups = _coordset_to_groups(coordset)
    rebuilt = _groups_to_coordset(groups, name=coordset.name)

    assert len(groups) == 1
    assert groups[0].dim == "x"
    assert len(groups[0].entries) == 1
    _assert_public_coordset_equal(coordset, rebuilt)
    _assert_public_coord_equal(coordset["x"], rebuilt["x"])


def test_multi_coordinate_group_conversion_preserves_aliases_and_metadata():
    coord_a = Coord(
        [1.0, 2.0, 3.0],
        name="a",
        title="alpha",
        units="cm^-1",
        labels=["u", "v", "w"],
    )
    coord_b = Coord([4.0, 5.0, 6.0], name="b", title="beta", units="s")
    coordset = CoordSet([coord_a, coord_b])

    groups = _coordset_to_groups(coordset)
    rebuilt = _groups_to_coordset(groups, name=coordset.name)

    assert len(groups) == 1
    group = groups[0]
    assert group.dim == "x"
    assert len(group.entries) == 2
    assert group.aliases["_1"] == group.entries[0].id
    assert group.aliases["_2"] == group.entries[1].id
    _assert_public_coord_equal(group.entries[0].coord, coordset["x"]["_1"])
    _assert_public_coord_equal(group.entries[1].coord, coordset["x"]["_2"])

    _assert_public_coordset_equal(coordset, rebuilt)
    _assert_public_coord_equal(coordset["x"]["_1"], rebuilt["x"]["_1"])
    _assert_public_coord_equal(coordset["x"]["_2"], rebuilt["x"]["_2"])
    _assert_public_coord_equal(coordset["x_1"], rebuilt["x_1"])
    _assert_public_coord_equal(coordset["_1"], rebuilt["_1"])


def test_non_first_default_roundtrip_preserves_selected_coordinate():
    coord_a = Coord([1.0, 2.0, 3.0], name="a", title="alpha", units="cm^-1")
    coord_b = Coord([4.0, 5.0, 6.0], name="b", title="beta", units="s")
    coordset = CoordSet([coord_a, coord_b])
    coordset["x"].select(2)

    groups = _coordset_to_groups(coordset)
    rebuilt = _groups_to_coordset(groups, name=coordset.name)

    assert groups[0].default_id == groups[0].entries[1].id
    _assert_public_coord_equal(coordset["x"].default, rebuilt["x"].default)
    assert_array_equal(coordset["x"].data, rebuilt["x"].data)


def test_reference_roundtrip_preserves_reference_key_and_lookup_behavior():
    coord = Coord([1.0, 2.0, 3.0], name="x", title="distance", units="cm")
    coordset = CoordSet(x=coord, y="x")

    groups = _coordset_to_groups(coordset)
    rebuilt = _groups_to_coordset(groups, name=coordset.name)

    assert len(groups) == 2
    assert groups[1].reference is not None
    assert groups[1].reference.dim == "y"
    assert groups[1].reference.target_dim == "x"
    assert rebuilt.references == coordset.references
    assert rebuilt["y"] == coordset["y"] == "x"
    _assert_public_coord_equal(coordset["x"], rebuilt["x"])


def test_public_roundtrip_equivalence_for_mixed_coordset_cases():
    coord_x = Coord([1.0, 2.0, 3.0], name="x", title="distance", units="cm")
    coord_a = Coord([10.0, 20.0, 30.0], name="a", title="alpha", units="s")
    coord_b = Coord([11.0, 21.0, 31.0], name="b", title="beta", units="ms")
    coordset = CoordSet(x=CoordSet(coord_a, coord_b), y=coord_x, z="y")
    coordset["x"].select(2)

    rebuilt = _groups_to_coordset(_coordset_to_groups(coordset), name=coordset.name)

    _assert_public_coordset_equal(coordset, rebuilt)
    _assert_public_coord_equal(coordset["x"].default, rebuilt["x"].default)
    _assert_public_coord_equal(coordset["x_1"], rebuilt["x_1"])
    _assert_public_coord_equal(coordset["_1"], rebuilt["_1"])
    _assert_public_coord_equal(coordset["y"], rebuilt["y"])
    assert rebuilt["z"] == coordset["z"] == "y"
