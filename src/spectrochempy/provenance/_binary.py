# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Bounded provenance for binary arithmetic between two datasets."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any

from spectrochempy.provenance._identity import ObservedState
from spectrochempy.provenance._instrument import BINARY_ADD_OPERATION_ID
from spectrochempy.provenance._instrument import BINARY_MULTIPLY_OPERATION_ID
from spectrochempy.provenance._instrument import BINARY_SUBTRACT_OPERATION_ID
from spectrochempy.provenance._instrument import BINARY_TRUE_DIVIDE_OPERATION_ID
from spectrochempy.provenance._models import OperationRecord
from spectrochempy.provenance._models import ReferenceLink

if TYPE_CHECKING:
    from spectrochempy.provenance._capture import ProvenanceCapture

_PROVIDER_NAME = "spectrochempy"
_OPERATION_IDS = {
    "add": BINARY_ADD_OPERATION_ID,
    "sub": BINARY_SUBTRACT_OPERATION_ID,
    "subtract": BINARY_SUBTRACT_OPERATION_ID,
    "mul": BINARY_MULTIPLY_OPERATION_ID,
    "multiply": BINARY_MULTIPLY_OPERATION_ID,
    "truediv": BINARY_TRUE_DIVIDE_OPERATION_ID,
    "divide": BINARY_TRUE_DIVIDE_OPERATION_ID,
    "true_divide": BINARY_TRUE_DIVIDE_OPERATION_ID,
}


@dataclass(frozen=True, slots=True)
class BinaryBoundary:
    """Ordered input states observed before binary arithmetic executes."""

    capture: ProvenanceCapture
    left: ObservedState
    right: ObservedState
    started_at: datetime


def operation_id(name: str) -> str | None:
    """Return the P3 semantic operation identifier for *name*, if supported."""
    return _OPERATION_IDS.get(name)


def is_dataset_pair(inputs: tuple[Any, ...] | list[Any]) -> bool:
    """Return whether *inputs* is exactly two NDDataset operands."""
    if len(inputs) != 2:
        return False
    from spectrochempy.core.dataset.nddataset import NDDataset  # noqa: PLC0415

    return all(isinstance(value, NDDataset) for value in inputs)


def prepare_boundary(
    capture: ProvenanceCapture,
    left: Any,
    right: Any,
    *,
    operation_id: str,
    started_at: datetime | None,
) -> BinaryBoundary | None:
    """Observe both ordered inputs without allowing capture failure to escape."""
    try:
        left_state = capture._observe_source(left)
        right_state = capture._observe_source(right)
        return BinaryBoundary(
            capture=capture,
            left=left_state,
            right=right_state,
            started_at=started_at or datetime.now(UTC),
        )
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, exc)
        with suppress(Exception):
            capture._invalidate_state(left)
        with suppress(Exception):
            capture._invalidate_state(right)
        return None


def _capture_mapping(boundary: BinaryBoundary) -> dict[str, Any] | None:
    omissions = []
    if boundary.left.changed:
        omissions.append(
            {
                "field": "inputs[0].reference",
                "status": "unknown",
                "reason": "unrecorded_state_change",
            }
        )
    if boundary.right.changed:
        omissions.append(
            {
                "field": "inputs[1].reference",
                "status": "unknown",
                "reason": "unrecorded_state_change",
            }
        )
    if not omissions:
        return None
    return {"status": "partial", "omissions": omissions}


def _record_kwargs(boundary: BinaryBoundary) -> dict[str, Any]:
    capture_mapping = _capture_mapping(boundary)
    return {"capture": capture_mapping} if capture_mapping is not None else {}


def _failure_warning(exc: BaseException) -> tuple[dict[str, str], ...]:
    cls = type(exc)
    return (
        {
            "status": "capture",
            "reason": "operation_failed",
            "exception_type": f"{cls.__module__}.{cls.__qualname__}",
        },
    )


def invalidate_unrecorded_result(
    capture: ProvenanceCapture,
    result: Any,
    *,
    operation_id: str,
) -> None:
    """Break result continuity after a binary-boundary capture failure."""
    try:
        capture._invalidate_state(result)
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, exc)


def record_success(
    boundary: BinaryBoundary,
    result: Any,
    *,
    operation_id: str,
    implementation: str,
    requested_parameters: dict[str, Any],
    dispatch: str,
) -> None:
    """Record one successful out-of-place binary dataset operation."""
    capture = boundary.capture
    try:
        output_state = capture._register_output(result)
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=operation_id,
            category="combine",
            implementation=implementation,
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("left", boundary.left.state),
                ReferenceLink("right", boundary.right.state),
            ),
            outputs=(ReferenceLink("result", output_state),),
            requested_parameters=requested_parameters,
            resolved_parameters={
                "dispatch": dispatch,
                "result_shape": list(result.shape),
                "result_dims": list(result.dims),
            },
            summary="Combined two NDDataset operands",
            strict=False,
            **_record_kwargs(boundary),
        )
        capture.ledger.append(record)
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, exc)
        invalidate_unrecorded_result(capture, result, operation_id=operation_id)


def record_failure(
    boundary: BinaryBoundary,
    exc: BaseException,
    *,
    operation_id: str,
    implementation: str,
    requested_parameters: dict[str, Any],
    dispatch: str,
) -> None:
    """Record a failed non-mutating binary operation without masking *exc*."""
    capture = boundary.capture
    try:
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=operation_id,
            category="combine",
            implementation=implementation,
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("left", boundary.left.state),
                ReferenceLink("right", boundary.right.state),
            ),
            outputs=(),
            requested_parameters=requested_parameters,
            resolved_parameters={"dispatch": dispatch},
            status="failed",
            warnings=_failure_warning(exc),
            summary="Binary NDDataset operation failed",
            strict=False,
            **_record_kwargs(boundary),
        )
        capture.ledger.append(record)
    except Exception as capture_exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, capture_exc)
