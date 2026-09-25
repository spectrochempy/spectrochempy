# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Bounded direct-call provenance for :class:`CenterTransformer`."""

from __future__ import annotations

import hashlib
import math
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from pint import Unit

from spectrochempy.provenance._identity import ObservedState
from spectrochempy.provenance._instrument import CENTER_FIT_OPERATION_ID
from spectrochempy.provenance._instrument import CENTER_TRANSFORM_OPERATION_ID
from spectrochempy.provenance._models import OperationRecord
from spectrochempy.provenance._models import ReferenceLink

if TYPE_CHECKING:
    from spectrochempy.processing.transformation.preprocessing_transformers import (
        CenterTransformer,
    )
    from spectrochempy.provenance._capture import ProvenanceCapture

_PROVIDER_NAME = "spectrochempy"
_IMPLEMENTATION = (
    "spectrochempy.processing.transformation.preprocessing_transformers."
    "CenterTransformer"
)


@dataclass(frozen=True, slots=True)
class CenterBoundary:
    """Capture-time references observed before a Center operation executes."""

    capture: ProvenanceCapture
    transformer: ObservedState
    dataset: ObservedState
    transformer_fingerprint: bytes | None
    started_at: datetime


def _hash_value(digest: Any, value: Any) -> bool:
    """Hash a restricted runtime value without using arbitrary representations."""
    if value is None:
        digest.update(b"none;")
        return True
    if isinstance(value, bool):
        digest.update(b"bool:1;" if value else b"bool:0;")
        return True
    if isinstance(value, int | np.integer):
        digest.update(f"int:{int(value)};".encode("ascii"))
        return True
    if isinstance(value, float | np.floating):
        number = float(value)
        if math.isnan(number):
            text = "nan"
        elif math.isinf(number):
            text = "+inf" if number > 0 else "-inf"
        else:
            text = number.hex()
        digest.update(f"float:{text};".encode("ascii"))
        return True
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        return True
    if isinstance(value, Unit):
        return _hash_value(digest, format(value, "~"))
    if isinstance(value, np.ndarray | np.ma.MaskedArray):
        array = np.asanyarray(value)
        if array.dtype.hasobject:
            return False
        data = np.ascontiguousarray(np.ma.getdata(array))
        mask = np.ascontiguousarray(np.ma.getmaskarray(array), dtype=np.bool_)
        digest.update(f"array:{data.shape}:{data.dtype};".encode("ascii"))
        digest.update(memoryview(data))
        digest.update(memoryview(mask))
        return True
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            return False
        digest.update(f"dict:{len(value)};".encode("ascii"))
        for key in sorted(value):
            if not _hash_value(digest, key) or not _hash_value(digest, value[key]):
                return False
        return True
    if isinstance(value, list | tuple):
        digest.update(f"sequence:{len(value)};".encode("ascii"))
        return all(_hash_value(digest, item) for item in value)
    return False


def center_transformer_fingerprint(transformer: CenterTransformer) -> bytes | None:
    """Return a digest of Center configuration and learned runtime state."""
    digest = hashlib.blake2b(digest_size=16)
    state = {
        "dim": transformer.dim,
        "fitted": transformer._fitted,
        "dim_name": getattr(transformer, "_dim_name", None),
        "fit_signature": getattr(transformer, "_fit_signature_", None),
        "mean": getattr(transformer, "mean_", None),
    }
    try:
        if not _hash_value(digest, state):
            return None
    except (TypeError, ValueError, BufferError):
        return None
    return digest.digest()


def prepare_boundary(
    capture: ProvenanceCapture,
    transformer: CenterTransformer,
    dataset: Any,
    *,
    operation_id: str,
    started_at: datetime | None,
) -> CenterBoundary | None:
    """Observe operation inputs without allowing capture failure to escape."""
    try:
        fingerprint = center_transformer_fingerprint(transformer)
        transformer_state = capture._observe_state(transformer, fingerprint)
        dataset_state = capture._observe_source(dataset)
        return CenterBoundary(
            capture=capture,
            transformer=transformer_state,
            dataset=dataset_state,
            transformer_fingerprint=fingerprint,
            started_at=started_at or datetime.now(UTC),
        )
    except Exception as exc:  # noqa: BLE001
        capture._note_capture_warning(operation_id, exc)
        capture._invalidate_state(transformer)
        return None


def invalidate_unrecorded_result(capture: ProvenanceCapture, result: Any) -> None:
    """Break result continuity after a transform boundary capture failure."""
    try:
        capture._invalidate_state(result)
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(CENTER_TRANSFORM_OPERATION_ID, exc)


def _capture_mapping(
    boundary: CenterBoundary,
    *,
    extra_omissions: tuple[dict[str, str], ...] = (),
) -> dict[str, Any] | None:
    omissions = list(extra_omissions)
    if boundary.transformer.changed:
        omissions.append(
            {
                "field": "inputs[0].reference",
                "status": "unknown",
                "reason": "unrecorded_state_change",
            }
        )
    if boundary.transformer_fingerprint is None:
        omissions.append(
            {
                "field": "inputs[0].reference",
                "status": "unknown",
                "reason": "transformer_state_unverifiable",
            }
        )
    if boundary.dataset.changed:
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


def _record_kwargs(
    boundary: CenterBoundary,
    *,
    extra_omissions: tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    capture_mapping = _capture_mapping(
        boundary,
        extra_omissions=extra_omissions,
    )
    return {"capture": capture_mapping} if capture_mapping is not None else {}


def _resolved_parameters(
    transformer: CenterTransformer, dataset: Any
) -> dict[str, Any]:
    axis, dim_name = dataset.get_axis(transformer.dim)
    return {
        "dim": transformer.dim,
        "axis": axis,
        "dimension": str(dim_name),
    }


def _learned_state_summary(transformer: CenterTransformer) -> dict[str, Any]:
    mean = np.ma.asarray(transformer.mean_)
    return {
        "fitted": bool(transformer._fitted),
        "dimension": str(transformer._dim_name),
        "mean": {
            "status": "omitted",
            "reason": "learned_array_values",
            "dtype": str(mean.dtype),
            "shape": list(mean.shape),
            "size": int(mean.size),
            "masked": int(np.ma.count_masked(mean)),
        },
    }


def _failure_warning(exc: BaseException) -> tuple[dict[str, str], ...]:
    cls = type(exc)
    return (
        {
            "status": "capture",
            "reason": "operation_failed",
            "exception_type": f"{cls.__module__}.{cls.__qualname__}",
        },
    )


def record_fit_success(
    boundary: CenterBoundary,
    transformer: CenterTransformer,
    dataset: Any,
) -> None:
    """Record a successful fit and advance one transformer identity."""
    capture = boundary.capture
    try:
        fingerprint = center_transformer_fingerprint(transformer)
        fitted_state = capture._advance_state(
            transformer,
            boundary.transformer.state,
            fingerprint,
        )
        extra = ()
        if fingerprint is None:
            extra = (
                {
                    "field": "outputs[0].reference",
                    "status": "unknown",
                    "reason": "transformer_state_unverifiable",
                },
            )
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=CENTER_FIT_OPERATION_ID,
            category="fit",
            implementation=f"{_IMPLEMENTATION}.fit",
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("transformer", boundary.transformer.state),
                ReferenceLink("calibration", boundary.dataset.state),
            ),
            outputs=(ReferenceLink("fitted_transformer", fitted_state),),
            requested_parameters={},
            resolved_parameters=_resolved_parameters(transformer, dataset),
            fitted_state_summary=_learned_state_summary(transformer),
            summary="Fitted CenterTransformer",
            strict=False,
            **_record_kwargs(boundary, extra_omissions=extra),
        )
        capture.ledger.append(record)
    except Exception as exc:  # noqa: BLE001
        capture._note_capture_warning(CENTER_FIT_OPERATION_ID, exc)
        capture._invalidate_state(transformer)


def record_fit_failure(
    boundary: CenterBoundary,
    transformer: CenterTransformer,
    dataset: Any,
    exc: BaseException,
) -> None:
    """Record a failed fit and its observed effect on transformer state."""
    capture = boundary.capture
    try:
        current = center_transformer_fingerprint(transformer)
        extra: tuple[dict[str, str], ...] = ()
        if boundary.transformer_fingerprint is None or current is None:
            effect = "unknown"
            capture._invalidate_state(transformer)
            extra = (
                {
                    "field": "state_effect",
                    "status": "unknown",
                    "reason": "transformer_state_unverifiable",
                },
            )
        elif current == boundary.transformer_fingerprint:
            effect = "unchanged"
        else:
            effect = "changed"
            capture._advance_state(
                transformer,
                boundary.transformer.state,
                current,
            )
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=CENTER_FIT_OPERATION_ID,
            category="fit",
            implementation=f"{_IMPLEMENTATION}.fit",
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("transformer", boundary.transformer.state),
                ReferenceLink("calibration", boundary.dataset.state),
            ),
            outputs=(),
            requested_parameters={},
            resolved_parameters={"dim": transformer.dim},
            status="failed",
            state_effect={"effect": effect, "reconstructable": False},
            warnings=_failure_warning(exc),
            summary="CenterTransformer fit failed",
            strict=False,
            **_record_kwargs(boundary, extra_omissions=extra),
        )
        capture.ledger.append(record)
    except Exception as capture_exc:  # noqa: BLE001
        capture._note_capture_warning(CENTER_FIT_OPERATION_ID, capture_exc)
        capture._invalidate_state(transformer)


def record_transform_success(
    boundary: CenterBoundary,
    transformer: CenterTransformer,
    dataset: Any,
    result: Any,
) -> None:
    """Record a successful transform without retaining scientific objects."""
    capture = boundary.capture
    try:
        current = center_transformer_fingerprint(transformer)
        extra: tuple[dict[str, str], ...] = ()
        if current is None or boundary.transformer_fingerprint is None:
            capture._invalidate_state(transformer)
        elif current != boundary.transformer_fingerprint:
            capture._invalidate_state(transformer)
            extra = (
                {
                    "field": "inputs[0].reference",
                    "status": "unknown",
                    "reason": "transformer_state_changed_during_transform",
                },
            )
        output_state = capture._register_output(result)
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=CENTER_TRANSFORM_OPERATION_ID,
            category="transform",
            implementation=f"{_IMPLEMENTATION}.transform",
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("transformer", boundary.transformer.state),
                ReferenceLink("source", boundary.dataset.state),
            ),
            outputs=(ReferenceLink("result", output_state),),
            requested_parameters={},
            resolved_parameters=_resolved_parameters(transformer, dataset),
            summary="Applied fitted CenterTransformer",
            strict=False,
            **_record_kwargs(boundary, extra_omissions=extra),
        )
        capture.ledger.append(record)
    except Exception as exc:  # noqa: BLE001
        capture._note_capture_warning(CENTER_TRANSFORM_OPERATION_ID, exc)
        capture._invalidate_state(result)


def record_transform_failure(
    boundary: CenterBoundary,
    transformer: CenterTransformer,
    dataset: Any,
    exc: BaseException,
) -> None:
    """Record a failed transform while preserving the scientific exception."""
    capture = boundary.capture
    try:
        current = center_transformer_fingerprint(transformer)
        extra: tuple[dict[str, str], ...] = ()
        if boundary.transformer_fingerprint is None or current is None:
            effect = "unknown"
            capture._invalidate_state(transformer)
            extra = (
                {
                    "field": "state_effect",
                    "status": "unknown",
                    "reason": "transformer_state_unverifiable",
                },
            )
        elif current == boundary.transformer_fingerprint:
            effect = "unchanged"
        else:
            effect = "changed"
            capture._advance_state(
                transformer,
                boundary.transformer.state,
                current,
            )
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=CENTER_TRANSFORM_OPERATION_ID,
            category="transform",
            implementation=f"{_IMPLEMENTATION}.transform",
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("transformer", boundary.transformer.state),
                ReferenceLink("source", boundary.dataset.state),
            ),
            outputs=(),
            requested_parameters={},
            resolved_parameters={"dim": transformer.dim},
            status="failed",
            state_effect={"effect": effect, "reconstructable": False},
            warnings=_failure_warning(exc),
            summary="CenterTransformer transform failed",
            strict=False,
            **_record_kwargs(boundary, extra_omissions=extra),
        )
        capture.ledger.append(record)
    except Exception as capture_exc:  # noqa: BLE001
        capture._note_capture_warning(CENTER_TRANSFORM_OPERATION_ID, capture_exc)
        capture._invalidate_state(transformer)
