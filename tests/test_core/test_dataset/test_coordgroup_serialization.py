# SPDX-License-Identifier: BSD-3-Clause
# (see LICENSE.txt for details)

"""Focused tests for private CoordSet serializer adapter helpers."""

from spectrochempy.core.dataset._coordgroup import _groups_to_legacy_state
from spectrochempy.core.dataset._coordgroup import _legacy_state_to_groups
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.utils.jsonutils import json_encoder


def _legacy_state(coordset):
    return json_encoder(coordset, encoding="base64")


def test_legacy_state_roundtrip_for_simple_coord_dimension():
    coordset = CoordSet(
        x=Coord([1.0, 2.0, 3.0], name="x", title="distance", units="cm"),
    )
    state = _legacy_state(coordset)

    groups = _legacy_state_to_groups(state)
    rebuilt = _groups_to_legacy_state(
        groups,
        name=state["name"],
        is_same_dim=state["is_same_dim"],
    )

    assert rebuilt == state


def test_legacy_state_roundtrip_for_multi_coordinate_dimension():
    coord_a = Coord([1.0, 2.0, 3.0], name="a", title="alpha", units="cm^-1")
    coord_b = Coord([4.0, 5.0, 6.0], name="b", title="beta", units="s")
    coordset = CoordSet([coord_a, coord_b])
    coordset["x"].select(2)
    state = _legacy_state(coordset)

    groups = _legacy_state_to_groups(state)
    rebuilt = _groups_to_legacy_state(
        groups,
        name=state["name"],
        is_same_dim=state["is_same_dim"],
    )

    assert rebuilt == state
    assert rebuilt["coords"][0]["default_index"] == 1
    assert rebuilt["coords"][0]["coords"][0]["name"] == "_1"
    assert rebuilt["coords"][0]["coords"][1]["name"] == "_2"


def test_legacy_state_roundtrip_for_reference_dimension():
    coordset = CoordSet(
        x=Coord([1.0, 2.0, 3.0], name="x", title="distance", units="cm"),
        y="x",
    )
    state = _legacy_state(coordset)

    groups = _legacy_state_to_groups(state)
    rebuilt = _groups_to_legacy_state(
        groups,
        name=state["name"],
        is_same_dim=state["is_same_dim"],
    )

    assert rebuilt == state
    assert rebuilt["references"] == {"y": "x"}


def test_legacy_state_without_default_index_uses_legacy_fallback():
    coord_a = Coord([1.0, 2.0, 3.0], name="a", title="alpha", units="cm^-1")
    coord_b = Coord([4.0, 5.0, 6.0], name="b", title="beta", units="s")
    coordset = CoordSet([coord_a, coord_b])
    coordset["x"].select(2)
    state = _legacy_state(coordset)
    state["coords"][0].pop("default_index", None)

    groups = _legacy_state_to_groups(state)
    rebuilt = _groups_to_legacy_state(
        groups,
        name=state["name"],
        is_same_dim=state["is_same_dim"],
    )

    assert groups[0].default_id == groups[0].entries[0].id
    assert rebuilt["coords"][0]["default_index"] == 0
