# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Immutable structured provenance value models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from types import MappingProxyType
from typing import Any
from typing import ClassVar

from spectrochempy.provenance._normalization import normalize_value

OPERATION_RECORD_SCHEMA_ID = "urn:spectrochempy:schema:operation-record:0.1"
OPERATION_RECORD_SCHEMA_VERSION = "0.1"

_CATEGORIES = {
    "read",
    "transform",
    "combine",
    "fit",
    "predict",
    "analyze",
    "export",
    "plot",
}
_STATUSES = {"succeeded", "failed"}
_CAPTURE_STATUSES = {"complete", "partial", "unknown"}
_STATE_EFFECTS = {"unchanged", "changed", "unknown"}


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _incomplete_paths(value: Any, path: str = "parameters") -> list[dict[str, str]]:
    omissions = []
    if isinstance(value, Mapping):
        status = value.get("status")
        if status in {"unknown", "omitted", "redacted", "unsupported"}:
            omissions.append({"field": path, "status": status})
        if value.get("redaction") is not None:
            omissions.append({"field": path, "status": "redacted"})
        for key, item in value.items():
            omissions.extend(_incomplete_paths(item, f"{path}.{key}"))
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            omissions.extend(_incomplete_paths(item, f"{path}[{index}]"))
    return omissions


def _utc_timestamp(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include timezone information")
    return value.astimezone(UTC)


def _timestamp_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _discover_provider_version(provider_name: str, provider_version: str | None):
    if provider_version is not None:
        return provider_version
    try:
        return version(provider_name)
    except PackageNotFoundError:
        return None


def _parameter_field_statuses(values: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for name, value in values.items():
        if isinstance(value, Mapping) and value.get("status") in {
            "unknown",
            "omitted",
            "redacted",
            "unsupported",
        }:
            fields[name] = {"status": value["status"]}
        elif isinstance(value, Mapping) and value.get("redaction") is not None:
            fields[name] = {"status": "redacted"}
    return fields


def _merge_omissions(*groups: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_encoding = {}
    for omission in (item for group in groups for item in group):
        detached = _thaw(omission)
        encoding = json.dumps(
            detached,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        by_encoding[encoding] = detached
    return [by_encoding[key] for key in sorted(by_encoding)]


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """
    Opaque identity of one logical runtime object within a ledger.

    Parameters
    ----------
    id : str
        Ledger-local object identifier.
    """

    id: str

    def __post_init__(self):
        _validate_identifier(self.id, "ObjectRef.id")


@dataclass(frozen=True, slots=True)
class StateRef:
    """
    Reference to one observed version of an object state.

    Parameters
    ----------
    id : str
        Ledger-local state identifier.
    object : ObjectRef
        Logical object whose state was observed.
    version : int
        Non-negative state version within the logical object.
    """

    id: str
    object: ObjectRef
    version: int

    kind: ClassVar[str] = "state"

    def __post_init__(self):
        _validate_identifier(self.id, "StateRef.id")
        if not isinstance(self.object, ObjectRef):
            raise TypeError("StateRef.object must be an ObjectRef")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("StateRef.version must be an integer")
        if self.version < 0:
            raise ValueError("StateRef.version must be non-negative")


@dataclass(frozen=True, slots=True)
class OperationRef:
    """
    Reference to an operation record within the same ledger.

    Parameters
    ----------
    id : str
        Ledger-local operation-record identifier.
    """

    id: str

    kind: ClassVar[str] = "operation"

    def __post_init__(self):
        _validate_identifier(self.id, "OperationRef.id")


@dataclass(frozen=True, slots=True)
class ResultRef:
    """
    Reference to a bounded non-dataset result descriptor.

    Parameters
    ----------
    id : str
        Ledger-local result identifier.
    """

    id: str

    kind: ClassVar[str] = "result"

    def __post_init__(self):
        _validate_identifier(self.id, "ResultRef.id")


@dataclass(frozen=True, slots=True)
class ReferenceLink:
    """
    Role-labelled state or result reference used by an operation record.

    Parameters
    ----------
    role : str
        Semantic role of the referenced value in the operation.
    reference : StateRef or ResultRef
        Referenced state or bounded result.
    """

    role: str
    reference: StateRef | ResultRef

    def __post_init__(self):
        _validate_identifier(self.role, "ReferenceLink.role")
        if not isinstance(self.reference, StateRef | ResultRef):
            raise TypeError("ReferenceLink.reference must be a StateRef or ResultRef")

    def to_dict(self) -> dict[str, str]:
        """Return a detached JSON-compatible reference entry."""
        return {
            "role": self.role,
            "kind": self.reference.kind,
            "id": self.reference.id,
        }


@dataclass(frozen=True, slots=True)
class OperationRecord:
    """
    Immutable description of one observed semantic operation.

    Parameters are normalized through the conservative provenance value
    policy. Explicit construction is strict by default; pass ``strict=False``
    to replace unsupported values with visible structured markers.

    Parameters
    ----------
    id : str
        Ledger-local record identifier.
    operation_id : str
        Stable namespaced semantic operation identifier.
    category : str
        Controlled operation category.
    implementation : str or None, optional
        Fully qualified implementation identifier used only for diagnostics.
    provider_name : str
        Stable provider or distribution name.
    provider_version : str or None
        Provider version. If omitted, local distribution metadata is queried
        and an explicit unknown marker is emitted when unavailable.
    started_at : datetime
        Timezone-aware operation start time.
    inputs : tuple of ReferenceLink, optional
        Ordered role-labelled input references.
    outputs : tuple of ReferenceLink, optional
        Ordered role-labelled output references.
    requested_parameters : mapping, optional
        Parameters explicitly requested at the semantic boundary.
    resolved_parameters : mapping or None, optional
        Effective public parameters known after validation.
    ended_at : datetime or None, optional
        Timezone-aware operation end time.
    status : {"succeeded", "failed"}, optional
        Scientific operation outcome.
    state_effect : mapping or None, optional
        Effect of a failed potentially mutating operation.
    fitted_state_summary : mapping or None, optional
        Bounded fitted-state description or omission markers.
    capture : mapping, optional
        Capture completeness and structured omissions.
    warnings : tuple of mappings, optional
        Bounded structured warnings.
    summary : str or None, optional
        Human-readable text that is never parsed for semantics.
    parent_operation : OperationRef or None, optional
        Genuine parent operation for nested execution.
    strict : bool, optional
        Whether unsupported parameter values raise during construction.
    """

    id: str
    operation_id: str
    category: str
    provider_name: str
    started_at: datetime
    provider_version: str | None = None
    implementation: str | None = None
    inputs: tuple[ReferenceLink, ...] = ()
    outputs: tuple[ReferenceLink, ...] = ()
    requested_parameters: Mapping[str, Any] = field(default_factory=dict)
    resolved_parameters: Mapping[str, Any] | None = None
    ended_at: datetime | None = None
    status: str = "succeeded"
    state_effect: Mapping[str, Any] | None = None
    fitted_state_summary: Mapping[str, Any] | None = None
    capture: Mapping[str, Any] = field(
        default_factory=lambda: {"status": "complete", "omissions": []}
    )
    warnings: tuple[Mapping[str, Any], ...] = ()
    summary: str | None = None
    parent_operation: OperationRef | None = None
    strict: bool = field(default=True, repr=False, compare=False)

    schema_id: ClassVar[str] = OPERATION_RECORD_SCHEMA_ID
    schema_version: ClassVar[str] = OPERATION_RECORD_SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be a boolean")
        _validate_identifier(self.id, "OperationRecord.id")
        _validate_identifier(self.operation_id, "OperationRecord.operation_id")
        if "." not in self.operation_id:
            raise ValueError("OperationRecord.operation_id must be namespaced")
        if self.category not in _CATEGORIES:
            raise ValueError(f"Unsupported operation category: {self.category}")
        if self.implementation is not None:
            _validate_identifier(self.implementation, "OperationRecord.implementation")
            if "." not in self.implementation:
                raise ValueError(
                    "OperationRecord.implementation must be fully qualified"
                )
        _validate_identifier(self.provider_name, "OperationRecord.provider_name")
        provider_version = _discover_provider_version(
            self.provider_name,
            self.provider_version,
        )
        if provider_version is not None:
            _validate_identifier(
                provider_version,
                "OperationRecord.provider_version",
            )
        if self.status not in _STATUSES:
            raise ValueError(f"Unsupported operation status: {self.status}")

        started_at = _utc_timestamp(self.started_at, "started_at")
        ended_at = self.ended_at
        if ended_at is not None:
            ended_at = _utc_timestamp(ended_at, "ended_at")
            if ended_at < started_at:
                raise ValueError("ended_at must not precede started_at")

        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if not all(isinstance(item, ReferenceLink) for item in inputs + outputs):
            raise TypeError("inputs and outputs must contain ReferenceLink values")
        if self.status == "failed" and outputs:
            raise ValueError("A failed operation record cannot contain outputs")
        if self.status == "succeeded" and not outputs:
            raise ValueError("A succeeded operation record must contain outputs")

        if not isinstance(self.requested_parameters, Mapping):
            raise TypeError("requested_parameters must be a mapping")
        if self.resolved_parameters is not None and not isinstance(
            self.resolved_parameters, Mapping
        ):
            raise TypeError("resolved_parameters must be a mapping or None")
        requested = normalize_value(
            self.requested_parameters,
            strict=self.strict,
        )
        resolved = (
            None
            if self.resolved_parameters is None
            else normalize_value(self.resolved_parameters, strict=self.strict)
        )
        if not isinstance(self.capture, Mapping):
            raise TypeError("capture must be a mapping")
        capture = normalize_value(self.capture, strict=True)
        capture_status = capture.get("status")
        if capture_status not in _CAPTURE_STATUSES:
            raise ValueError(f"Unsupported capture status: {capture_status}")
        capture_omissions = capture.get("omissions", [])
        if not isinstance(capture_omissions, list) or any(
            not isinstance(item, Mapping) for item in capture_omissions
        ):
            raise ValueError("capture.omissions must be a list of mappings")
        omissions = _incomplete_paths(requested, "parameters.requested")
        if resolved is not None:
            omissions.extend(_incomplete_paths(resolved, "parameters.resolved"))
        if omissions:
            capture = {
                **capture,
                "status": "partial" if capture_status == "complete" else capture_status,
                "omissions": _merge_omissions(capture_omissions, omissions),
            }
        if any(not isinstance(warning, Mapping) for warning in self.warnings):
            raise TypeError("warnings must contain mappings")
        warnings = tuple(
            normalize_value(warning, strict=self.strict) for warning in self.warnings
        )
        if self.state_effect is not None and not isinstance(self.state_effect, Mapping):
            raise TypeError("state_effect must be a mapping or None")
        state_effect = (
            None
            if self.state_effect is None
            else normalize_value(self.state_effect, strict=True)
        )
        if state_effect is not None:
            effect = state_effect.get("effect")
            if effect not in _STATE_EFFECTS:
                raise ValueError(f"Unsupported state effect: {effect}")
            reconstructable = state_effect.get("reconstructable")
            if not isinstance(reconstructable, bool):
                raise ValueError("state_effect.reconstructable must be a boolean")
        if self.fitted_state_summary is not None and not isinstance(
            self.fitted_state_summary, Mapping
        ):
            raise TypeError("fitted_state_summary must be a mapping or None")
        fitted_state_summary = (
            None
            if self.fitted_state_summary is None
            else normalize_value(self.fitted_state_summary, strict=self.strict)
        )
        if self.category == "fit" and self.status == "succeeded":
            fitted_omissions = (
                [{"field": "fitted_state_summary", "status": "omitted"}]
                if fitted_state_summary is None
                else _incomplete_paths(
                    fitted_state_summary,
                    "fitted_state_summary",
                )
            )
            if fitted_omissions:
                if capture["status"] == "complete":
                    capture["status"] = "partial"
                capture["omissions"] = _merge_omissions(
                    capture.get("omissions", []),
                    fitted_omissions,
                )
        if self.parent_operation is not None and not isinstance(
            self.parent_operation, OperationRef
        ):
            raise TypeError("parent_operation must be an OperationRef")
        if self.parent_operation is not None and self.parent_operation.id == self.id:
            raise ValueError("parent_operation cannot reference the record itself")
        if self.summary is not None and not isinstance(self.summary, str):
            raise TypeError("summary must be a string or None")

        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "provider_version", provider_version)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "requested_parameters", _freeze(requested))
        object.__setattr__(self, "resolved_parameters", _freeze(resolved))
        object.__setattr__(self, "capture", _freeze(capture))
        object.__setattr__(self, "warnings", tuple(_freeze(item) for item in warnings))
        object.__setattr__(self, "state_effect", _freeze(state_effect))
        object.__setattr__(
            self,
            "fitted_state_summary",
            _freeze(fitted_state_summary),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation of this record."""
        provider_version: Any = self.provider_version
        if provider_version is None:
            provider_version = {"status": "unknown", "reason": "not_discoverable"}
        requested_values = _thaw(self.requested_parameters)
        parameters: dict[str, Any] = {
            "requested": {
                "values": requested_values,
                "fields": _parameter_field_statuses(requested_values),
            }
        }
        if self.resolved_parameters is not None:
            resolved_values = _thaw(self.resolved_parameters)
            parameters["resolved"] = {
                "values": resolved_values,
                "fields": _parameter_field_statuses(resolved_values),
            }
        result = {
            "schema": {"id": self.schema_id, "version": self.schema_version},
            "id": self.id,
            "operation": {
                "id": self.operation_id,
                "category": self.category,
            },
            "provider": {
                "name": self.provider_name,
                "version": provider_version,
            },
            "started_at": _timestamp_text(self.started_at),
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "parameters": parameters,
            "status": self.status,
            "capture": _thaw(self.capture),
        }
        if self.implementation is not None:
            result["operation"]["implementation"] = self.implementation
        if self.ended_at is not None:
            result["ended_at"] = _timestamp_text(self.ended_at)
        if self.state_effect is not None:
            result["state_effect"] = _thaw(self.state_effect)
        if self.category == "fit" and self.status == "succeeded":
            result["fitted_state_summary"] = (
                {
                    "status": "omitted",
                    "reason": "not_captured",
                }
                if self.fitted_state_summary is None
                else _thaw(self.fitted_state_summary)
            )
        if self.warnings:
            result["warnings"] = _thaw(self.warnings)
        if self.summary is not None:
            result["summary"] = self.summary
        if self.parent_operation is not None:
            result["parent_operation"] = {
                "kind": self.parent_operation.kind,
                "id": self.parent_operation.id,
            }
        return result
