# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Private CoordSet group-model helpers used to prepare storage migration."""

from __future__ import annotations

import copy as cpy
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils.jsonutils import json_decoder
from spectrochempy.utils.jsonutils import json_encoder

if TYPE_CHECKING:
    from spectrochempy.core.dataset.coordset import CoordSet


@dataclass(frozen=True)
class _CoordinateEntry:
    """One coordinate entry inside a future dimension-coordinate group."""

    id: str
    coord: Coord
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CoordReference:
    """Reference from one dimension name to another."""

    dim: str
    target_dim: str


@dataclass(frozen=True)
class _DimensionCoordinates:
    """Future internal representation for all coordinates of one dimension."""

    dim: str
    entries: tuple[_CoordinateEntry, ...]
    default_id: str | None = None
    aliases: dict[str, str] = field(default_factory=dict)
    reference: _CoordReference | None = None


def _make_entry_id(coord: Coord, fallback: str, used_ids: set[str]) -> str:
    """Create a deterministic private entry id for one coordinate."""
    base = coord.title or coord.name or fallback
    entry_id = base
    suffix = 2

    while entry_id in used_ids:
        entry_id = f"{base}#{suffix}"
        suffix += 1

    used_ids.add(entry_id)
    return entry_id


def _coordset_group_to_dimension(coordset: CoordSet) -> _DimensionCoordinates:
    """Convert a same-dimension CoordSet compatibility group to one private group."""
    used_ids: set[str] = set()
    aliases: dict[str, str] = {}
    entries: list[_CoordinateEntry] = []

    for index, coord in enumerate(coordset.coords, start=1):
        alias = (
            coordset.names[index - 1]
            if index - 1 < len(coordset.names)
            else f"_{index}"
        )
        entry_id = _make_entry_id(coord, alias, used_ids)
        entries.append(
            _CoordinateEntry(
                id=entry_id,
                coord=coord.copy(keepname=True),
                aliases=(alias,),
            )
        )
        aliases[alias] = entry_id

    default_id = None
    if entries:
        default_index = min(max(0, coordset._default), len(entries) - 1)
        default_id = entries[default_index].id

    reference = None
    if coordset.references:
        ref_dim, target_dim = next(iter(coordset.references.items()))
        reference = _CoordReference(dim=ref_dim, target_dim=target_dim)

    return _DimensionCoordinates(
        dim=coordset.name,
        entries=tuple(entries),
        default_id=default_id,
        aliases=aliases,
        reference=reference,
    )


def _coordset_to_groups(coordset: CoordSet) -> tuple[_DimensionCoordinates, ...]:
    """Convert a legacy CoordSet into private dimension-coordinate groups."""
    if coordset.is_same_dim:
        return (_coordset_group_to_dimension(coordset),)

    groups: list[_DimensionCoordinates] = []

    for coord in coordset.coords:
        if coord is None:
            continue

        if coord._implements("CoordSet"):
            groups.append(_coordset_group_to_dimension(coord))
            continue

        entry = _CoordinateEntry(
            id=_make_entry_id(coord, coord.name, set()),
            coord=coord.copy(keepname=True),
        )
        groups.append(
            _DimensionCoordinates(
                dim=coord.name,
                entries=(entry,),
                default_id=entry.id,
            )
        )

    for dim, target_dim in coordset.references.items():
        groups.append(
            _DimensionCoordinates(
                dim=dim,
                entries=(),
                reference=_CoordReference(dim=dim, target_dim=target_dim),
            )
        )

    return tuple(groups)


def _groups_to_coordset(
    groups: tuple[_DimensionCoordinates, ...] | list[_DimensionCoordinates],
    *,
    name: str | None = None,
) -> CoordSet:
    """Convert private dimension-coordinate groups back to the legacy CoordSet model."""
    from spectrochempy.core.dataset.coordset import CoordSet

    coords = []
    references = {}

    for group in groups:
        if group.reference is not None:
            references[group.reference.dim] = group.reference.target_dim
            continue

        if len(group.entries) == 1 and not group.aliases:
            coord = group.entries[0].coord.copy(keepname=True)
            coord.name = group.dim
            coords.append(coord)
            continue

        children = [entry.coord.copy(keepname=True) for entry in group.entries]
        nested = CoordSet(*children, name=group.dim, sorted=False)
        nested._is_same_dim = True

        alias_names = []
        for index, entry in enumerate(group.entries, start=1):
            alias = next(
                (item for item in entry.aliases if group.aliases.get(item) == entry.id),
                None,
            )
            alias_names.append(alias or f"_{index}")

        nested._set_names(alias_names)
        nested._set_parent_dim(group.dim)

        if group.entries:
            default_index = next(
                (
                    index
                    for index, entry in enumerate(group.entries)
                    if entry.id == group.default_id
                ),
                0,
            )
            nested._default = default_index

        coords.append(nested)

    coordset = CoordSet(*coords, keepnames=True, sorted=False)
    if name is not None:
        coordset.name = name
    coordset._references = references
    return coordset


def _coord_from_legacy_state(state: dict) -> Coord:
    """Rebuild one Coord from legacy serialized state."""
    coord = Coord()
    for key, value in state.items():
        if key == "__class__":
            continue
        decoded = (
            json_decoder(cpy.deepcopy(value), allow_unsafe_legacy=True)
            if isinstance(value, dict)
            else value
        )
        setattr(coord, key if key == "name" else f"_{key}", decoded)
    return coord


def _coord_to_legacy_state(coord: Coord) -> dict:
    """Serialize one Coord to the legacy coord-state shape."""
    return json_encoder(coord, encoding="base64")


def _legacy_group_state_to_dimension(state: dict) -> _DimensionCoordinates:
    """Convert one legacy same-dimension serialized state to a private group."""
    used_ids: set[str] = set()
    aliases: dict[str, str] = {}
    entries: list[_CoordinateEntry] = []

    for index, coord_state in enumerate(state.get("coords", ()), start=1):
        coord = _coord_from_legacy_state(coord_state)
        alias = coord_state.get("name", f"_{index}")
        entry_id = _make_entry_id(coord, alias, used_ids)
        entries.append(
            _CoordinateEntry(
                id=entry_id,
                coord=coord,
                aliases=(alias,),
            )
        )
        aliases[alias] = entry_id

    default_index = state.get("default_index")
    if default_index is None and isinstance(state.get("default"), int):
        default_index = state["default"]

    default_id = None
    if entries:
        index = min(max(0, int(default_index or 0)), len(entries) - 1)
        default_id = entries[index].id

    reference = None
    references = state.get("references", {})
    if references:
        ref_dim, target_dim = next(iter(references.items()))
        reference = _CoordReference(dim=ref_dim, target_dim=target_dim)

    return _DimensionCoordinates(
        dim=state["name"],
        entries=tuple(entries),
        default_id=default_id,
        aliases=aliases,
        reference=reference,
    )


def _legacy_state_to_groups(state: dict) -> tuple[_DimensionCoordinates, ...]:
    """Convert legacy serialized CoordSet state to private groups."""
    if state.get("is_same_dim"):
        return (_legacy_group_state_to_dimension(state),)

    groups: list[_DimensionCoordinates] = []

    for item in state.get("coords", ()):
        if "coords" in item:
            groups.append(_legacy_group_state_to_dimension(item))
            continue

        coord = _coord_from_legacy_state(item)
        entry = _CoordinateEntry(
            id=_make_entry_id(coord, item["name"], set()),
            coord=coord,
        )
        groups.append(
            _DimensionCoordinates(
                dim=item["name"],
                entries=(entry,),
                default_id=entry.id,
            )
        )

    for dim, target_dim in state.get("references", {}).items():
        groups.append(
            _DimensionCoordinates(
                dim=dim,
                entries=(),
                reference=_CoordReference(dim=dim, target_dim=target_dim),
            )
        )

    return tuple(groups)


def _groups_to_legacy_state(
    groups: tuple[_DimensionCoordinates, ...] | list[_DimensionCoordinates],
    *,
    name: str,
    is_same_dim: bool = False,
) -> dict:
    """Convert private groups back to the legacy serialized CoordSet state."""
    if is_same_dim:
        if len(groups) != 1:
            raise ValueError("Same-dimension legacy state expects exactly one group")
        group = groups[0]
        child_states = []
        for index, entry in enumerate(group.entries, start=1):
            coord_state = _coord_to_legacy_state(entry.coord)
            alias = next(
                (item for item in entry.aliases if group.aliases.get(item) == entry.id),
                None,
            )
            coord_state["name"] = alias or f"_{index}"
            child_states.append(coord_state)

        default_index = 0
        if group.entries:
            default_index = next(
                (
                    index
                    for index, entry in enumerate(group.entries)
                    if entry.id == group.default_id
                ),
                0,
            )

        return {
            "coords": child_states,
            "references": {},
            "is_same_dim": True,
            "default_index": default_index,
            "name": name,
        }

    references: dict[str, str] = {}
    coords: list[dict] = []

    for group in groups:
        if group.reference is not None:
            references[group.reference.dim] = group.reference.target_dim
            continue

        if len(group.entries) == 1 and not group.aliases and not is_same_dim:
            coord_state = _coord_to_legacy_state(group.entries[0].coord)
            coord_state["name"] = group.dim
            coords.append(coord_state)
            continue

        child_states = []
        for index, entry in enumerate(group.entries, start=1):
            coord_state = _coord_to_legacy_state(entry.coord)
            alias = next(
                (item for item in entry.aliases if group.aliases.get(item) == entry.id),
                None,
            )
            coord_state["name"] = alias or f"_{index}"
            child_states.append(coord_state)

        default_index = 0
        if group.entries:
            default_index = next(
                (
                    index
                    for index, entry in enumerate(group.entries)
                    if entry.id == group.default_id
                ),
                0,
            )

        coords.append(
            {
                "coords": child_states,
                "references": {},
                "is_same_dim": True,
                "default_index": default_index,
                "name": group.dim,
            }
        )

    return {
        "coords": coords,
        "references": references,
        "is_same_dim": is_same_dim,
        "default_index": 0,
        "name": name,
    }
