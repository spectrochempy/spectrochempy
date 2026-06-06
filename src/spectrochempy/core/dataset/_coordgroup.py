# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Private CoordSet group-model helpers used to prepare storage migration."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from spectrochempy.core.dataset.coord import Coord

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
        alias = coordset.names[index - 1] if index - 1 < len(coordset.names) else f"_{index}"
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
                (
                    item
                    for item in entry.aliases
                    if group.aliases.get(item) == entry.id
                ),
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
