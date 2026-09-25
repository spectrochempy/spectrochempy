# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Best-effort helpers shared by the bounded provenance runtime slices.

Out-of-place dataset selection and transpose use the record helpers in this
module. Direct Center lifecycle capture lives in :mod:`._center`, and bounded
binary dataset arithmetic capture lives in :mod:`._binary`; both reuse the
context-local boundary and operation-specific suppression defined here. Capture
code never alters scientific values, mutates scientific arguments, or masks a
scientific exception.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any

from spectrochempy.provenance._identity import ObservedState
from spectrochempy.provenance._models import OperationRecord
from spectrochempy.provenance._models import ReferenceLink

if TYPE_CHECKING:
    from spectrochempy.provenance._capture import ProvenanceCapture

PROVIDER_NAME = "spectrochempy"
SLICE_OPERATION_ID = "org.spectrochempy.dataset.slice"
TRANSPOSE_OPERATION_ID = "org.spectrochempy.dataset.transpose"
CENTER_FIT_OPERATION_ID = "org.spectrochempy.preprocessing.center.fit"
CENTER_TRANSFORM_OPERATION_ID = "org.spectrochempy.preprocessing.center.transform"
BINARY_ADD_OPERATION_ID = "org.spectrochempy.dataset.binary.add"
BINARY_SUBTRACT_OPERATION_ID = "org.spectrochempy.dataset.binary.subtract"
BINARY_MULTIPLY_OPERATION_ID = "org.spectrochempy.dataset.binary.multiply"
BINARY_TRUE_DIVIDE_OPERATION_ID = "org.spectrochempy.dataset.binary.true_divide"
BINARY_OPERATION_IDS = (
    BINARY_ADD_OPERATION_ID,
    BINARY_SUBTRACT_OPERATION_ID,
    BINARY_MULTIPLY_OPERATION_ID,
    BINARY_TRUE_DIVIDE_OPERATION_ID,
)

_SUPPRESSED_OPERATIONS: ContextVar[frozenset[str]] = ContextVar(
    "spectrochempy_suppressed_provenance_operations",
    default=frozenset(),
)

_UNRECORDED_CHANGE_REASON = "unrecorded_state_change"


@contextmanager
def suppress_provenance(*operation_ids: str):
    """Suppress selected operation families in this execution context."""
    suppressed = _SUPPRESSED_OPERATIONS.get()
    token = _SUPPRESSED_OPERATIONS.set(suppressed.union(operation_ids))
    try:
        yield
    finally:
        _SUPPRESSED_OPERATIONS.reset(token)


def provenance_boundary(
    operation_id: str | None = None,
) -> tuple[ProvenanceCapture | None, datetime | None]:
    """Return the active capture and a boundary timestamp, if any."""
    from spectrochempy.provenance._capture import ProvenanceCapture  # noqa: PLC0415

    if operation_id is not None and operation_id in _SUPPRESSED_OPERATIONS.get():
        return None, None
    capture = ProvenanceCapture.current()
    if capture is None:
        return None, None
    return capture, datetime.now(UTC)


def describe_selection(items: Any) -> Any:
    """Return a JSON-safe description of a selection used for indexing."""
    if isinstance(items, slice):
        return {
            "type": "slice",
            "start": items.start,
            "stop": items.stop,
            "step": items.step,
        }
    if items is Ellipsis:
        return {"type": "ellipsis"}
    if isinstance(items, tuple):
        return [describe_selection(item) for item in items]
    if isinstance(items, list):
        return [describe_selection(item) for item in items]
    return items


def _exception_identifier(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_requested_parameters(
    capture: ProvenanceCapture,
    operation_id: str,
    requested_parameters: dict[str, Any] | Callable[[], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Resolve requested parameters inside capture protection."""
    if callable(requested_parameters):
        try:
            return requested_parameters()
        except Exception as exc:  # noqa: BLE001
            capture._note_capture_warning(operation_id, exc)
            return {
                "status": "omitted",
                "reason": "parameter_description_failed",
            }
    return dict(requested_parameters or {})


def _capture_after_observed(observed: ObservedState) -> dict[str, Any] | None:
    """Return a partial capture descriptor when an unrecorded change is detected."""
    if not observed.changed:
        return None
    return {
        "status": "partial",
        "omissions": [
            {
                "field": "inputs[0].reference",
                "status": "unknown",
                "reason": _UNRECORDED_CHANGE_REASON,
            }
        ],
    }


def record_out_of_place(
    capture: ProvenanceCapture,
    *,
    operation_id: str,
    source: Any,
    result: Any,
    requested_parameters: dict[str, Any] | Callable[[], dict[str, Any]],
    implementation: str,
    started_at: datetime | None = None,
    resolved_parameters: dict[str, Any] | None = None,
    summary: str | None = None,
) -> None:
    """Attempt to append one successful out-of-place operation record."""
    try:
        parameters = _resolve_requested_parameters(
            capture,
            operation_id,
            requested_parameters,
        )
        observed = capture._observe_source(source)
        input_state = observed.state
        output_state = capture._register_output(result)
        capture_mapping = _capture_after_observed(observed)
        kwargs: dict[str, Any] = (
            {"capture": capture_mapping} if capture_mapping is not None else {}
        )
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=operation_id,
            category="transform",
            implementation=implementation,
            provider_name=PROVIDER_NAME,
            started_at=started_at or datetime.now(UTC),
            ended_at=datetime.now(UTC),
            inputs=(ReferenceLink("source", input_state),),
            outputs=(ReferenceLink("result", output_state),),
            requested_parameters=parameters,
            resolved_parameters=resolved_parameters,
            summary=summary,
            strict=False,
            **kwargs,
        )
        capture.ledger.append(record)
    except Exception as exc:  # noqa: BLE001
        capture._note_capture_warning(operation_id, exc)


def record_failure(
    capture: ProvenanceCapture,
    *,
    operation_id: str,
    implementation: str,
    source: Any,
    exc: BaseException,
    requested_parameters: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    started_at: datetime | None = None,
) -> None:
    """Attempt to append one bounded failed record without masking *exc*."""
    try:
        parameters = _resolve_requested_parameters(
            capture,
            operation_id,
            requested_parameters,
        )
        observed = capture._observe_source(source)
        input_state = observed.state
        capture_mapping = _capture_after_observed(observed)
        kwargs: dict[str, Any] = (
            {"capture": capture_mapping} if capture_mapping is not None else {}
        )
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=operation_id,
            category="transform",
            implementation=implementation,
            provider_name=PROVIDER_NAME,
            started_at=started_at or datetime.now(UTC),
            ended_at=datetime.now(UTC),
            inputs=(ReferenceLink("source", input_state),),
            outputs=(),
            requested_parameters=parameters,
            status="failed",
            warnings=(
                {
                    "status": "capture",
                    "reason": "operation_failed",
                    "exception_type": _exception_identifier(exc),
                },
            ),
            summary=f"Operation failed with {_exception_identifier(exc)}",
            strict=False,
            **kwargs,
        )
        capture.ledger.append(record)
    except Exception as capture_exc:  # noqa: BLE001
        capture._note_capture_warning(operation_id, capture_exc)
