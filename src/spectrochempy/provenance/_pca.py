# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Bounded provenance for a direct PCA fit and score transform cycle."""

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
from numpy.random import RandomState

from spectrochempy.provenance._identity import ObservedState
from spectrochempy.provenance._instrument import PCA_FIT_OPERATION_ID
from spectrochempy.provenance._instrument import PCA_TRANSFORM_OPERATION_ID
from spectrochempy.provenance._models import OperationRecord
from spectrochempy.provenance._models import ReferenceLink

if TYPE_CHECKING:
    from spectrochempy.analysis.decomposition.pca import PCA
    from spectrochempy.provenance._capture import ProvenanceCapture

_PROVIDER_NAME = "spectrochempy"
_IMPLEMENTATION = "spectrochempy.analysis.decomposition.pca.PCA"
_CONFIGURATION_NAMES = (
    "iterated_power",
    "n_components",
    "n_oversamples",
    "power_iteration_normalizer",
    "random_state",
    "scaled",
    "standardized",
    "svd_solver",
    "tol",
    "whiten",
)
_BACKEND_STATE_NAMES = (
    "_fit_svd_solver",
    "components_",
    "explained_variance_",
    "explained_variance_ratio_",
    "mean_",
    "n_components_",
    "n_features_in_",
    "n_samples_",
    "noise_variance_",
    "singular_values_",
)


@dataclass(frozen=True, slots=True)
class PCABoundary:
    """References and configuration observed before a direct PCA operation."""

    capture: ProvenanceCapture
    estimator: ObservedState
    dataset: ObservedState
    estimator_fingerprint: bytes | None
    configuration: dict[str, Any]
    started_at: datetime


def _optional_attribute(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except Exception:  # noqa: BLE001
        return None


def _record_value(value: Any) -> Any:
    """Return a bounded JSON-safe configuration value."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, RandomState):
        return {
            "type": "numpy.random.RandomState",
            "state": {
                "status": "omitted",
                "reason": "random_state_internal_state",
            },
        }
    return {
        "status": "omitted",
        "reason": "unsupported_configuration_value",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def configuration_snapshot(estimator: PCA) -> dict[str, Any]:
    """Describe only PCA configuration that can affect fit or transform."""
    return {
        name: _record_value(getattr(estimator, name)) for name in _CONFIGURATION_NAMES
    }


def requested_transform_parameters(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Retain only explicitly supplied public transform parameters."""
    return (
        {"n_components": _record_value(kwargs["n_components"])}
        if "n_components" in kwargs
        else {}
    )


def _hash_value(digest: Any, value: Any) -> bool:
    """Hash a restricted value without arbitrary representations."""
    if value is None:
        digest.update(b"none;")
        return True
    if isinstance(value, bool | np.bool_):
        digest.update(b"bool:1;" if bool(value) else b"bool:0;")
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
    if isinstance(value, RandomState):
        return _hash_value(
            digest,
            {"type": "numpy.random.RandomState", "state": value.get_state()},
        )
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


def pca_fingerprint(estimator: PCA) -> bytes | None:
    """Hash PCA configuration and the bounded backend state used by transform."""
    backend = estimator._pca
    state = {
        "configuration": {
            name: getattr(estimator, name) for name in _CONFIGURATION_NAMES
        },
        "fitted": bool(estimator._fitted),
        "wrapper_n_components": _optional_attribute(estimator, "_n_components"),
        "backend": {
            name: _optional_attribute(backend, name) for name in _BACKEND_STATE_NAMES
        },
    }
    digest = hashlib.blake2b(digest_size=16)
    try:
        if not _hash_value(digest, state):
            return None
    except (TypeError, ValueError, BufferError):
        return None
    return digest.digest()


def prepare_boundary(
    capture: ProvenanceCapture,
    estimator: PCA,
    dataset: Any,
    *,
    operation_id: str,
    started_at: datetime | None,
) -> PCABoundary | None:
    """Observe a direct PCA operation without exposing capture failures."""
    try:
        configuration = configuration_snapshot(estimator)
        fingerprint = pca_fingerprint(estimator)
        estimator_state = capture._observe_state(estimator, fingerprint)
        dataset_state = capture._observe_source(dataset)
        return PCABoundary(
            capture=capture,
            estimator=estimator_state,
            dataset=dataset_state,
            estimator_fingerprint=fingerprint,
            configuration=configuration,
            started_at=started_at or datetime.now(UTC),
        )
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(operation_id, exc)
        with suppress(Exception):
            capture._invalidate_state(estimator)
        return None


def invalidate_unrecorded_result(
    capture: ProvenanceCapture,
    result: Any,
) -> None:
    """Break result continuity when a transform record cannot be written."""
    try:
        capture._invalidate_state(result)
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(PCA_TRANSFORM_OPERATION_ID, exc)


def _capture_mapping(
    boundary: PCABoundary,
    *,
    extra_omissions: tuple[dict[str, str], ...] = (),
) -> dict[str, Any] | None:
    omissions = list(extra_omissions)
    if boundary.estimator.changed:
        omissions.append(
            {
                "field": "inputs[0].reference",
                "status": "unknown",
                "reason": "unrecorded_state_change",
            }
        )
    if boundary.estimator_fingerprint is None:
        omissions.append(
            {
                "field": "inputs[0].reference",
                "status": "unknown",
                "reason": "estimator_state_unverifiable",
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
    boundary: PCABoundary,
    *,
    extra_omissions: tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    capture_mapping = _capture_mapping(
        boundary,
        extra_omissions=extra_omissions,
    )
    return {"capture": capture_mapping} if capture_mapping is not None else {}


def _requested_parameters(
    boundary: PCABoundary,
    call: dict[str, Any],
) -> dict[str, Any]:
    return {"configuration": boundary.configuration, "call": call}


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "status": "omitted",
        "reason": "learned_array_values",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "size": int(array.size),
    }


def _learned_state_summary(estimator: PCA) -> dict[str, Any]:
    backend = estimator._pca
    return {
        "fitted": bool(estimator._fitted),
        "backend": {
            "status": "omitted",
            "reason": "fitted_model_not_serialized",
            "type": f"{type(backend).__module__}.{type(backend).__qualname__}",
        },
        "n_components": int(backend.n_components_),
        "n_observations": int(backend.n_samples_),
        "n_features": int(backend.n_features_in_),
        "svd_solver": str(backend._fit_svd_solver),
        "noise_variance": float(backend.noise_variance_),
        "components": _array_summary(backend.components_),
        "mean": _array_summary(backend.mean_),
        "explained_variance": _array_summary(backend.explained_variance_),
        "explained_variance_ratio": _array_summary(backend.explained_variance_ratio_),
        "singular_values": _array_summary(backend.singular_values_),
    }


def _fit_resolved_parameters(estimator: PCA, dataset: Any) -> dict[str, Any]:
    backend = estimator._pca
    return {
        "configuration": {
            "n_components": int(backend.n_components_),
            "svd_solver": str(backend._fit_svd_solver),
            "scaled": bool(estimator.scaled),
            "standardized": bool(estimator.standardized),
            "whiten": bool(backend.whiten),
        },
        "input": {"shape": list(dataset.shape), "dims": list(dataset.dims)},
        "effective_input_shape": [
            int(backend.n_samples_),
            int(backend.n_features_in_),
        ],
    }


def _transform_resolved_parameters(
    estimator: PCA,
    dataset: Any,
    result: Any,
) -> dict[str, Any]:
    backend = estimator._pca
    return {
        "configuration": {
            "fitted_n_components": int(backend.n_components_),
            "svd_solver": str(backend._fit_svd_solver),
            "scaled": bool(estimator.scaled),
            "standardized": bool(estimator.standardized),
            "whiten": bool(backend.whiten),
        },
        "call": {"n_components": int(result.shape[-1])},
        "input": {"shape": list(dataset.shape), "dims": list(dataset.dims)},
        "result": {"shape": list(result.shape), "dims": list(result.dims)},
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
    boundary: PCABoundary,
    estimator: PCA,
    dataset: Any,
) -> None:
    """Record a successful fit and advance one PCA identity."""
    capture = boundary.capture
    try:
        fingerprint = pca_fingerprint(estimator)
        fitted_state = capture._advance_state(
            estimator,
            boundary.estimator.state,
            fingerprint,
        )
        extra = ()
        if fingerprint is None:
            extra = (
                {
                    "field": "outputs[0].reference",
                    "status": "unknown",
                    "reason": "estimator_state_unverifiable",
                },
            )
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=PCA_FIT_OPERATION_ID,
            category="fit",
            implementation=f"{_IMPLEMENTATION}.fit",
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("estimator", boundary.estimator.state),
                ReferenceLink("calibration", boundary.dataset.state),
            ),
            outputs=(ReferenceLink("fitted_estimator", fitted_state),),
            requested_parameters=_requested_parameters(boundary, {}),
            resolved_parameters=_fit_resolved_parameters(estimator, dataset),
            fitted_state_summary=_learned_state_summary(estimator),
            summary="Fitted PCA estimator",
            strict=False,
            **_record_kwargs(boundary, extra_omissions=extra),
        )
        capture.ledger.append(record)
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(PCA_FIT_OPERATION_ID, exc)
        with suppress(Exception):
            capture._invalidate_state(estimator)


def _state_effect_after_failure(
    boundary: PCABoundary,
    estimator: PCA,
) -> tuple[str, tuple[dict[str, str], ...]]:
    capture = boundary.capture
    current = pca_fingerprint(estimator)
    if boundary.estimator_fingerprint is None or current is None:
        capture._invalidate_state(estimator)
        return (
            "unknown",
            (
                {
                    "field": "state_effect",
                    "status": "unknown",
                    "reason": "estimator_state_unverifiable",
                },
            ),
        )
    if current == boundary.estimator_fingerprint:
        return "unchanged", ()
    capture._advance_state(estimator, boundary.estimator.state, current)
    return "changed", ()


def record_fit_failure(
    boundary: PCABoundary,
    estimator: PCA,
    dataset: Any,
    exc: BaseException,
) -> None:
    """Record a failed initial fit or refit and its estimator state effect."""
    capture = boundary.capture
    try:
        effect, extra = _state_effect_after_failure(boundary, estimator)
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=PCA_FIT_OPERATION_ID,
            category="fit",
            implementation=f"{_IMPLEMENTATION}.fit",
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("estimator", boundary.estimator.state),
                ReferenceLink("calibration", boundary.dataset.state),
            ),
            outputs=(),
            requested_parameters=_requested_parameters(boundary, {}),
            status="failed",
            state_effect={"effect": effect, "reconstructable": False},
            warnings=_failure_warning(exc),
            summary="PCA fit failed",
            strict=False,
            **_record_kwargs(boundary, extra_omissions=extra),
        )
        capture.ledger.append(record)
    except Exception as capture_exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(PCA_FIT_OPERATION_ID, capture_exc)
        with suppress(Exception):
            capture._invalidate_state(estimator)


def record_transform_success(
    boundary: PCABoundary,
    estimator: PCA,
    dataset: Any,
    result: Any,
    *,
    requested_call: dict[str, Any],
) -> None:
    """Record a direct score transform without retaining scientific objects."""
    capture = boundary.capture
    try:
        current = pca_fingerprint(estimator)
        extra: tuple[dict[str, str], ...] = ()
        if current is None or boundary.estimator_fingerprint is None:
            capture._invalidate_state(estimator)
            extra = (
                {
                    "field": "inputs[0].reference",
                    "status": "unknown",
                    "reason": "estimator_state_unverifiable",
                },
            )
        elif current != boundary.estimator_fingerprint:
            capture._invalidate_state(estimator)
            extra = (
                {
                    "field": "inputs[0].reference",
                    "status": "unknown",
                    "reason": "estimator_state_changed_during_transform",
                },
            )
        output_state = capture._register_output(result)
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=PCA_TRANSFORM_OPERATION_ID,
            category="transform",
            implementation=f"{_IMPLEMENTATION}.transform",
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("estimator", boundary.estimator.state),
                ReferenceLink("source", boundary.dataset.state),
            ),
            outputs=(ReferenceLink("result", output_state),),
            requested_parameters=_requested_parameters(boundary, requested_call),
            resolved_parameters=_transform_resolved_parameters(
                estimator,
                dataset,
                result,
            ),
            summary="Produced PCA scores",
            strict=False,
            **_record_kwargs(boundary, extra_omissions=extra),
        )
        capture.ledger.append(record)
    except Exception as exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(PCA_TRANSFORM_OPERATION_ID, exc)
        invalidate_unrecorded_result(capture, result)


def record_transform_failure(
    boundary: PCABoundary,
    estimator: PCA,
    dataset: Any,
    exc: BaseException,
    *,
    requested_call: dict[str, Any],
) -> None:
    """Record a failed transform without masking its scientific exception."""
    capture = boundary.capture
    try:
        effect, extra = _state_effect_after_failure(boundary, estimator)
        record = OperationRecord(
            id=capture._next_record_id(),
            operation_id=PCA_TRANSFORM_OPERATION_ID,
            category="transform",
            implementation=f"{_IMPLEMENTATION}.transform",
            provider_name=_PROVIDER_NAME,
            started_at=boundary.started_at,
            ended_at=datetime.now(UTC),
            inputs=(
                ReferenceLink("estimator", boundary.estimator.state),
                ReferenceLink("source", boundary.dataset.state),
            ),
            outputs=(),
            requested_parameters=_requested_parameters(boundary, requested_call),
            status="failed",
            state_effect={"effect": effect, "reconstructable": False},
            warnings=_failure_warning(exc),
            summary="PCA transform failed",
            strict=False,
            **_record_kwargs(boundary, extra_omissions=extra),
        )
        capture.ledger.append(record)
    except Exception as capture_exc:  # noqa: BLE001
        with suppress(Exception):
            capture._note_capture_warning(PCA_TRANSFORM_OPERATION_ID, capture_exc)
        with suppress(Exception):
            capture._invalidate_state(estimator)
