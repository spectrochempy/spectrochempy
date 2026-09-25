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

To avoid advertising known continuity when an object may have been mutated
outside the recorded operations, each registration stores a bounded
fingerprint for verification. Dataset callers use the default numeric-state
fingerprint; other bounded slices may supply a domain-specific digest. When a
previously observed object is observed again, its current fingerprint is
compared with the stored one: a mismatch means the object changed without a
recorded operation, so the registry creates a fresh unrecorded state for the
current content and reports the divergence instead of silently reusing the
stale state. Objects without a verifiable fingerprint are treated
conservatively for the same reason.
"""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from typing import Any

from spectrochempy.provenance._models import ObjectRef
from spectrochempy.provenance._models import StateRef

_DEFAULT_FINGERPRINT = object()


def _fingerprint(value: Any) -> bytes | None:
    """Return a bounded digest of the object's numeric state, if possible."""
    import numpy as np  # noqa: PLC0415

    buffer = getattr(value, "data", value)
    if not isinstance(buffer, np.ndarray):
        return None
    try:
        contiguous = np.ascontiguousarray(buffer)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(f"{contiguous.shape}:{contiguous.dtype}".encode("ascii"))
        digest.update(memoryview(contiguous))
        return digest.digest()
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class ObservedState:
    """
    Observed input state and whether it diverged from the registered one.

    Parameters
    ----------
    state : StateRef
        State reference to use for the consuming operation.
    changed : bool
        Whether the object's current content no longer matches the
        previously registered state and an unrecorded change is suspected.
    """

    state: StateRef
    changed: bool = False


class IdentityRegistry:
    """Weak, id-keyed association between live objects and ledger references."""

    def __init__(self):
        self.__entries: dict[
            int, tuple[weakref.ReferenceType | None, StateRef, bytes | None]
        ] = {}
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

    def _follow_observed_state(self, state_ref: StateRef) -> StateRef:
        self.__state_count += 1
        return StateRef(
            f"state-{self.__state_count:06d}",
            state_ref.object,
            version=state_ref.version + 1,
        )

    def observe(
        self,
        value: Any,
        *,
        fingerprint: bytes | None | object = _DEFAULT_FINGERPRINT,
    ) -> ObservedState:
        """Observe *value* as an input, verifying continuity when tracked."""
        key = id(value)
        current = (
            _fingerprint(value) if fingerprint is _DEFAULT_FINGERPRINT else fingerprint
        )
        entry = self.__entries.get(key)
        if entry is not None:
            reference, state_ref, recorded_fingerprint = entry
            if reference is not None and reference() is value:
                if recorded_fingerprint is not None and current == recorded_fingerprint:
                    return ObservedState(state_ref, changed=False)
                recorded = self._follow_observed_state(state_ref)
                self.__remember(key, value, recorded, fingerprint=current)
                return ObservedState(recorded, changed=True)

        object_ref = self._new_object_ref()
        state_ref = self._new_state_ref(object_ref)
        self.__remember(key, value, state_ref, fingerprint=current)
        return ObservedState(state_ref, changed=False)

    def advance(
        self,
        value: Any,
        state_ref: StateRef,
        *,
        fingerprint: bytes | None | object = _DEFAULT_FINGERPRINT,
    ) -> StateRef:
        """Advance a tracked logical object to its next observed state."""
        key = id(value)
        entry = self.__entries.get(key)
        if entry is None or entry[0] is None or entry[0]() is not value:
            raise ValueError("cannot advance an untracked object")
        if entry[1] != state_ref:
            raise ValueError("cannot advance an object from a stale state")
        current = (
            _fingerprint(value) if fingerprint is _DEFAULT_FINGERPRINT else fingerprint
        )
        advanced = self._follow_observed_state(state_ref)
        self.__remember(key, value, advanced, fingerprint=current)
        return advanced

    def invalidate(self, value: Any) -> None:
        """Make the next observation report uncertain state continuity."""
        key = id(value)
        entry = self.__entries.get(key)
        if entry is None:
            object_ref = self._new_object_ref()
            state_ref = self._new_state_ref(object_ref)
            self.__remember(key, value, state_ref, fingerprint=None)
            return
        reference, state_ref, _ = entry
        if reference is not None and reference() is value:
            self.__entries[key] = (reference, state_ref, None)

    def register_output(self, value: Any) -> StateRef:
        """Register a new logical object and its initial state for an output."""
        object_ref = self._new_object_ref()
        state_ref = self._new_state_ref(object_ref)
        self.__remember(id(value), value, state_ref)
        return state_ref

    def __remember(
        self,
        key: int,
        value: Any,
        state_ref: StateRef,
        *,
        fingerprint: bytes | None | object = _DEFAULT_FINGERPRINT,
    ) -> None:
        try:
            reference: weakref.ReferenceType | None = weakref.ref(value)
        except TypeError:
            reference = None
        current = (
            _fingerprint(value) if fingerprint is _DEFAULT_FINGERPRINT else fingerprint
        )
        self.__entries[key] = (reference, state_ref, current)

    def clear(self) -> None:
        """Release every live-object association."""
        self.__entries.clear()
