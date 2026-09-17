# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Transient identity registry for the experimental provenance substrate.

The registry associates live objects with ledger-local opaque references while
a :class:`~spectrochempy.provenance.ProvenanceCapture` is active. It never
retains a live object: entries are keyed by ``id()`` and hold only a weak
reference plus immutable reference values. Closing the context releases the
associations.
"""

from __future__ import annotations

import weakref
from typing import Any

from spectrochempy.provenance._models import ObjectRef
from spectrochempy.provenance._models import StateRef


class IdentityRegistry:
    """Weak, id-keyed association between live objects and ledger references."""

    def __init__(self):
        self.__entries: dict[int, tuple[weakref.ReferenceType | None, StateRef]] = {}
        self.__object_count = 0
        self.__state_count = 0

    def __len__(self) -> int:
        return len(self.__entries)

    def _new_object_ref(self) -> ObjectRef:
        self.__object_count += 1
        return ObjectRef(f"object-{self.__object_count:06d}")

    def _new_state_ref(self, object_ref: ObjectRef) -> StateRef:
        self.__state_count += 1
        return StateRef(f"state-{self.__state_count:06d}", object_ref, version=0)

    def observe(self, value: Any) -> StateRef:
        """Return the tracked current state, registering a new object if needed."""
        key = id(value)
        entry = self.__entries.get(key)
        if entry is not None:
            reference, state_ref = entry
            if reference is not None and reference() is value:
                return state_ref

        object_ref = self._new_object_ref()
        state_ref = self._new_state_ref(object_ref)
        self.__remember(key, value, state_ref)
        return state_ref

    def register_output(self, value: Any) -> StateRef:
        """Register a new logical object and its initial state for an output."""
        object_ref = self._new_object_ref()
        state_ref = self._new_state_ref(object_ref)
        self.__remember(id(value), value, state_ref)
        return state_ref

    def __remember(self, key: int, value: Any, state_ref: StateRef) -> None:
        try:
            reference: weakref.ReferenceType | None = weakref.ref(value)
        except TypeError:
            reference = None
        self.__entries[key] = (reference, state_ref)

    def clear(self) -> None:
        """Release every live-object association."""
        self.__entries.clear()
