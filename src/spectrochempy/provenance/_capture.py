# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Context-local provenance capture and append-only ledger."""

from __future__ import annotations

from contextvars import ContextVar
from contextvars import Token

from spectrochempy.provenance._models import ObjectRef
from spectrochempy.provenance._models import OperationRecord
from spectrochempy.provenance._models import OperationRef
from spectrochempy.provenance._models import ResultRef
from spectrochempy.provenance._models import StateRef

_ACTIVE_CAPTURE: ContextVar[ProvenanceCapture | None] = ContextVar(
    "spectrochempy_active_provenance_capture",
    default=None,
)


class ProvenanceLedger:
    """Append-only collection of immutable operation records."""

    def __init__(self):
        self.__operation_records: list[OperationRecord] = []
        self.__record_ids: set[str] = set()
        self.__identity_kinds: dict[str, str] = {}
        self.__objects: dict[str, ObjectRef] = {}
        self.__states: dict[str, StateRef] = {}
        self.__state_ids_by_version: dict[tuple[str, int], str] = {}
        self.__latest_state_versions: dict[str, int] = {}
        self.__results: dict[str, ResultRef] = {}
        self.__operation_ids: set[str] = set()

    @property
    def operation_records(self) -> tuple[OperationRecord, ...]:
        """Return an immutable snapshot of records in execution order."""
        return tuple(self.__operation_records)

    def append(self, record: OperationRecord) -> None:
        """Atomically append a validated record in execution order."""
        if not isinstance(record, OperationRecord):
            raise TypeError("record must be an OperationRecord")
        if record.id in self.__record_ids:
            raise ValueError(f"duplicate operation record id: {record.id}")

        identity_kinds = self.__identity_kinds.copy()
        objects = self.__objects.copy()
        states = self.__states.copy()
        state_ids_by_version = self.__state_ids_by_version.copy()
        latest_state_versions = self.__latest_state_versions.copy()
        results = self.__results.copy()
        operation_ids = self.__operation_ids.copy()

        def register_identity(identifier: str, kind: str) -> None:
            existing_kind = identity_kinds.get(identifier)
            if existing_kind is not None and existing_kind != kind:
                raise ValueError(
                    f"identifier {identifier!r} is already registered as "
                    f"{existing_kind}, not {kind}"
                )
            identity_kinds[identifier] = kind

        def register_object(reference: ObjectRef) -> None:
            register_identity(reference.id, "object")
            existing = objects.get(reference.id)
            if existing is not None and existing != reference:
                raise ValueError(f"conflicting object reference: {reference.id}")
            objects[reference.id] = reference

        def register_state(reference: StateRef) -> None:
            register_object(reference.object)
            register_identity(reference.id, "state")
            existing = states.get(reference.id)
            if existing is not None:
                if existing != reference:
                    raise ValueError(f"conflicting state reference: {reference.id}")
                return
            version_key = (reference.object.id, reference.version)
            existing_state_id = state_ids_by_version.get(version_key)
            if existing_state_id is not None and existing_state_id != reference.id:
                raise ValueError(
                    f"object {reference.object.id!r} version {reference.version} "
                    f"is already registered as state {existing_state_id!r}"
                )
            latest = latest_state_versions.get(reference.object.id)
            if latest is not None and reference.version <= latest:
                raise ValueError(
                    f"state version {reference.version} for object "
                    f"{reference.object.id!r} is not greater than {latest}"
                )
            states[reference.id] = reference
            state_ids_by_version[version_key] = reference.id
            latest_state_versions[reference.object.id] = reference.version

        def register_result(reference: ResultRef) -> None:
            register_identity(reference.id, "result")
            existing = results.get(reference.id)
            if existing is not None and existing != reference:
                raise ValueError(f"conflicting result reference: {reference.id}")
            results[reference.id] = reference

        def register_operation(reference: OperationRef) -> None:
            register_identity(reference.id, "operation")
            operation_ids.add(reference.id)

        def register_reference(reference: StateRef | ResultRef) -> None:
            if isinstance(reference, StateRef):
                register_state(reference)
            else:
                register_result(reference)

        register_operation(OperationRef(record.id))
        if record.parent_operation is not None:
            register_operation(record.parent_operation)
        for link in (*record.inputs, *record.outputs):
            register_reference(link.reference)

        self.__operation_records.append(record)
        self.__record_ids.add(record.id)
        self.__identity_kinds = identity_kinds
        self.__objects = objects
        self.__states = states
        self.__state_ids_by_version = state_ids_by_version
        self.__latest_state_versions = latest_state_versions
        self.__results = results
        self.__operation_ids = operation_ids

    def __len__(self) -> int:
        return len(self.__operation_records)

    def __iter__(self):
        return iter(self.operation_records)


class ProvenanceCapture:
    """
    Explicit context owning one context-local provenance ledger.

    P1 does not instrument scientific operations. Records are appended only
    through :attr:`ledger`; later phases may use :meth:`current` at verified
    semantic boundaries.
    """

    def __init__(self):
        self.__ledger = ProvenanceLedger()
        self.__token: Token | None = None
        self.__closed = False

    @property
    def ledger(self) -> ProvenanceLedger:
        """The append-only ledger owned by this capture."""
        return self.__ledger

    @property
    def active(self) -> bool:
        """Whether this capture is innermost in the current context."""
        return not self.__closed and _ACTIVE_CAPTURE.get() is self

    @classmethod
    def current(cls) -> ProvenanceCapture | None:
        """Return the innermost open capture in the current context, if any."""
        capture = _ACTIVE_CAPTURE.get()
        if capture is not None and capture.__closed:
            return None
        return capture

    def __enter__(self) -> ProvenanceCapture:
        if self.__closed:
            raise RuntimeError(
                "A ProvenanceCapture cannot be re-entered after it is closed"
            )
        if self.__token is not None:
            raise RuntimeError("A ProvenanceCapture cannot be re-entered")
        self.__token = _ACTIVE_CAPTURE.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        token = self.__token
        if token is None:
            raise RuntimeError("ProvenanceCapture context is not active")
        _ACTIVE_CAPTURE.reset(token)
        self.__token = None
        self.__closed = True
        return False
