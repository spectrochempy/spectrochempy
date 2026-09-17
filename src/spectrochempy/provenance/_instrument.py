# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Best-effort instrumentation helpers for the first provenance runtime slice.

Only out-of-place single-source dataset selection (slicing) and transpose are
instrumented. Capture code never alters scientific values, never mutates
scientific arguments, and never masks a scientific exception.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any

from spectrochempy.provenance._models import OperationRecord
from spectrochempy.provenance._models import ReferenceLink

if TYPE_CHECKING:
    from spectrochempy.provenance._capture import ProvenanceCapture

PROVIDER_NAME = "spectrochempy"
SLICE_OPERATION_ID = "org.spectrochempy.dataset.slice"
TRANSPOSE_OPERATION_ID = "org.spectrochempy.dataset.transpose"


def provenance_boundary() -> tuple[ProvenanceCapture | None, datetime | None]:
    """Return the active capture and a boundary timestamp, if any."""
    from spectrochempy.provenance._capture import ProvenanceCapture

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


def record_out_of_place(
    capture: ProvenanceCapture,
    *,
    operation_id: str,
    source: Any,
    result: Any,
    requested_parameters: dict[str, Any],
    implementation: str,
    started_at: datetime | None = None,
    resolved_parameters: dict[str, Any] | None = None,
    summary: str | None = None,
) -> None:
    """Attempt to append one successful out-of-place operation record."""
    try:
        input_state = capture._observe_source(source)
        output_state = capture._register_output(result)
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
            requested_parameters=requested_parameters,
            resolved_parameters=resolved_parameters,
            summary=summary,
            strict=False,
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
    started_at: datetime | None = None,
) -> None:
    """Attempt to append one bounded failed record without masking *exc*."""
    try:
        input_state = capture._observe_source(source)
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
            requested_parameters={},
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
        )
        capture.ledger.append(record)
    except Exception as capture_exc:  # noqa: BLE001
        capture._note_capture_warning(operation_id, capture_exc)
