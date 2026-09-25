# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Bounded provenance for ordered multi-dataset assembly."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any

from spectrochempy.provenance._identity import ObservedState
from spectrochempy.provenance._models import OperationRecord
from spectrochempy.provenance._models import ReferenceLink

if TYPE_CHECKING:
    from spectrochempy.provenance._capture import ProvenanceCapture

_PROVIDER_NAME = "spectrochempy"


@dataclass(frozen=True, slots=True)
class AssemblyBoundary:
    """Ordered input states observed before an assembly executes."""

    capture: ProvenanceCapture
    sources: tuple[ObservedState, ...]
    started_at: datetime


def is_dataset_sequence(values: list[Any]) -> bool:
    """Return whether every normalized input is an NDDataset."""
    if not values:
        return False
    from spectrochempy.core.dataset.nddataset import NDDataset  # noqa: PLC0415

    return all(isinstance(value, NDDataset) for value in values)


def prepare_boundary(
    capture: ProvenanceCapture,
    sources: list[Any],
    *,
    operation_id: str,
    started_at: datetime | None,
) -> AssemblyBoundary | None:
    """Observe each distinct input object once and retain every ordered role."""
    try:
        observed_by_identity: dict[int, tuple[Any, ObservedState]] = {}
        observed_sources = []
        for source in sources:
            entry = observed_by_identity.get(id(source))
            if entry is not None and entry[0] is source:
                observed = entry[1]
            else:
                observed = capture._observe_source(source)
                observed_by_identity[id(source)] = (source, observed)
            observed_sources.append(observed)
        return AssemblyBoundary(
            capture=capture,
            sources=tuple(observed_sources),
            started_at=started_at or datetime.now(UTC),
        )
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, exc)
        for source in sources:
            with suppress(Exception):
                capture._invalidate_state(source)
        return None


def _record_kwargs(boundary: AssemblyBoundary) -> dict[str, Any]:
    omissions = [
        {
            "field": f"inputs[{index}].reference",
            "status": "unknown",
            "reason": "unrecorded_state_change",
        }
        for index, observed in enumerate(boundary.sources)
        if observed.changed
    ]
    if not omissions:
        return {}
    return {"capture": {"status": "partial", "omissions": omissions}}


def _input_links(boundary: AssemblyBoundary) -> tuple[ReferenceLink, ...]:
    return tuple(
        ReferenceLink(f"source[{index}]", observed.state)
        for index, observed in enumerate(boundary.sources)
    )


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
    """Break result continuity after an assembly capture failure."""
    try:
        capture._invalidate_state(result)
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, exc)


def record_success(
    boundary: AssemblyBoundary,
    result: Any,
    *,
    operation_id: str,
    implementation: str,
    requested_parameters: dict[str, Any],
    resolved_parameters: dict[str, Any],
) -> None:
    """Record one successful out-of-place ordered assembly."""
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
            inputs=_input_links(boundary),
            outputs=(ReferenceLink("result", output_state),),
            requested_parameters=requested_parameters,
            resolved_parameters=resolved_parameters,
            summary=f"Combined {len(boundary.sources)} ordered NDDataset inputs",
            strict=False,
            **_record_kwargs(boundary),
        )
        capture.ledger.append(record)
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, exc)
        invalidate_unrecorded_result(capture, result, operation_id=operation_id)


def record_failure(
    boundary: AssemblyBoundary,
    exc: BaseException,
    *,
    operation_id: str,
    implementation: str,
    requested_parameters: dict[str, Any],
) -> None:
    """Record a failed non-mutating assembly without masking *exc*."""
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
            inputs=_input_links(boundary),
            outputs=(),
            requested_parameters=requested_parameters,
            status="failed",
            warnings=_failure_warning(exc),
            summary=f"Ordered assembly of {len(boundary.sources)} datasets failed",
            strict=False,
            **_record_kwargs(boundary),
        )
        capture.ledger.append(record)
    except Exception as capture_exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, capture_exc)
