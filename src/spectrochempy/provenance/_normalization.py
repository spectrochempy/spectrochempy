# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Restricted normalization for passive provenance values."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC
from datetime import date
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pint import Quantity
from pint import Unit

from spectrochempy.core.dataset.coord import Coord

DEFAULT_MAX_ENCODED_BYTES = 16 * 1024
MAX_COLLECTION_ITEMS = 256
MAX_NESTING_DEPTH = 32


class ProvenanceValidationError(ValueError):
    """Raised when a value violates the restricted provenance contract."""


def _type_identifier(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _status(status: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, **details}


def _unsupported(value: Any, *, strict: bool, reason: str) -> dict[str, Any]:
    if strict:
        raise ProvenanceValidationError(
            f"Unsupported provenance value type: {_type_identifier(value)}"
        )
    return _status(
        "unsupported",
        reason,
        type=_type_identifier(value),
    )


def _encoded_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _unit_text(unit: Unit) -> str:
    return format(unit, "~")


def _coordinate_range(value: Coord, *, strict: bool, max_encoded_bytes: int):
    if value.size == 0 or value.data is None:
        return None
    data = np.asarray(value.data)
    if data.dtype.kind not in "biufUSMm":
        if strict:
            raise ProvenanceValidationError(
                "Coordinate ranges require a safely representable dtype"
            )
        return _status("omitted", "coordinate_range_dtype", dtype=str(data.dtype))
    return [
        normalize_value(
            data.flat[0],
            strict=strict,
            max_encoded_bytes=max_encoded_bytes,
        ),
        normalize_value(
            data.flat[-1],
            strict=strict,
            max_encoded_bytes=max_encoded_bytes,
        ),
    ]


def normalize_value(
    value: Any,
    *,
    strict: bool = False,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    """
    Return a deterministic JSON-compatible representation of *value*.

    Unsupported values become an explicit status marker in best-effort mode.
    Strict mode raises :class:`ProvenanceValidationError` instead. Arbitrary
    object representations are never used.
    """
    if max_encoded_bytes < 1:
        raise ValueError("max_encoded_bytes must be a positive integer")
    if _depth > MAX_NESTING_DEPTH:
        if strict:
            raise ProvenanceValidationError("Provenance value nesting is too deep")
        return _status("omitted", "maximum_nesting_depth")

    if value is None or isinstance(value, bool):
        normalized = value
    elif isinstance(value, Enum):
        normalized = {
            "type": "enum",
            "enum": _type_identifier(value),
            "member": value.name,
            "value": normalize_value(
                value.value,
                strict=strict,
                max_encoded_bytes=max_encoded_bytes,
                _depth=_depth + 1,
                _seen=_seen,
            ),
        }
    elif isinstance(value, int):
        normalized = value
    elif isinstance(value, float):
        if math.isnan(value):
            normalized = {"type": "float", "value": "nan"}
        elif math.isinf(value):
            normalized = {
                "type": "float",
                "value": "+infinity" if value > 0 else "-infinity",
            }
        else:
            normalized = value
    elif isinstance(value, str):
        normalized = value
    elif isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, complex):
            return _unsupported(value, strict=strict, reason="complex_numpy_scalar")
        normalized = normalize_value(
            scalar,
            strict=strict,
            max_encoded_bytes=max_encoded_bytes,
            _depth=_depth + 1,
            _seen=_seen,
        )
    elif isinstance(value, Path):
        normalized = {
            "type": "path",
            "value": value.name,
            "redaction": "basename",
        }
    elif isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            if strict:
                raise ProvenanceValidationError(
                    "Provenance datetimes must include timezone information"
                )
            return _status("unsupported", "naive_datetime", type="datetime")
        utc_value = value.astimezone(UTC)
        normalized = {
            "type": "datetime",
            "value": utc_value.isoformat().replace("+00:00", "Z"),
        }
    elif isinstance(value, date):
        normalized = {"type": "date", "value": value.isoformat()}
    elif isinstance(value, Unit):
        normalized = {"type": "unit", "value": _unit_text(value)}
    elif isinstance(value, Quantity):
        normalized = {
            "type": "quantity",
            "magnitude": normalize_value(
                value.magnitude,
                strict=strict,
                max_encoded_bytes=max_encoded_bytes,
                _depth=_depth + 1,
                _seen=_seen,
            ),
            "unit": _unit_text(value.units),
        }
    elif isinstance(value, Coord):
        normalized = {
            "type": "coordinate",
            "dimension": value.dims[0] if value.dims else None,
            "title": value.title,
            "unit": _unit_text(value.units) if value.units is not None else None,
            "size": value.size,
            "range": _coordinate_range(
                value,
                strict=strict,
                max_encoded_bytes=max_encoded_bytes,
            ),
        }
        if value.labels is not None:
            normalized["labels"] = _status("omitted", "privacy_default")
    elif isinstance(value, np.ma.MaskedArray) and np.ma.is_masked(value):
        if strict:
            raise ProvenanceValidationError(
                "Masked array parameters are not embedded in provenance records"
            )
        normalized = {
            "status": "omitted",
            "reason": "masked_array",
            "type": "numpy.ma.MaskedArray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "size": int(value.size),
            "masked": int(np.ma.count_masked(value)),
        }
    elif isinstance(value, np.ndarray):
        if value.size > 64:
            if strict:
                raise ProvenanceValidationError(
                    "Provenance arrays cannot contain more than 64 elements"
                )
            normalized = {
                "status": "omitted",
                "reason": "array_size_limit",
                "type": "numpy.ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "size": int(value.size),
            }
        elif value.dtype.kind in "biufUS":
            normalized = {
                "type": "numpy.ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": normalize_value(
                    value.tolist(),
                    strict=strict,
                    max_encoded_bytes=max_encoded_bytes,
                    _depth=_depth + 1,
                    _seen=_seen,
                ),
            }
        else:
            return _unsupported(value, strict=strict, reason="numpy_dtype")
    elif isinstance(value, Mapping):
        _seen = set() if _seen is None else _seen
        if id(value) in _seen:
            if strict:
                raise ProvenanceValidationError("Cyclic provenance value")
            return _status("unsupported", "cyclic_value", type=_type_identifier(value))
        if len(value) > MAX_COLLECTION_ITEMS:
            if strict:
                raise ProvenanceValidationError("Provenance mapping is too large")
            return _status("omitted", "collection_size_limit", size=len(value))
        if any(not isinstance(key, str) for key in value):
            if strict:
                raise ProvenanceValidationError(
                    "Provenance mapping keys must be strings"
                )
            return _status("unsupported", "non_string_mapping_key")
        _seen.add(id(value))
        try:
            normalized = {
                key: normalize_value(
                    value[key],
                    strict=strict,
                    max_encoded_bytes=max_encoded_bytes,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
                for key in sorted(value)
            }
        finally:
            _seen.remove(id(value))
    elif isinstance(value, (list, tuple)):
        _seen = set() if _seen is None else _seen
        if id(value) in _seen:
            if strict:
                raise ProvenanceValidationError("Cyclic provenance value")
            return _status("unsupported", "cyclic_value", type=_type_identifier(value))
        if len(value) > MAX_COLLECTION_ITEMS:
            if strict:
                raise ProvenanceValidationError("Provenance collection is too large")
            return _status("omitted", "collection_size_limit", size=len(value))
        _seen.add(id(value))
        try:
            normalized = [
                normalize_value(
                    item,
                    strict=strict,
                    max_encoded_bytes=max_encoded_bytes,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
                for item in value
            ]
        finally:
            _seen.remove(id(value))
    elif isinstance(value, (set, frozenset)):
        if len(value) > MAX_COLLECTION_ITEMS:
            if strict:
                raise ProvenanceValidationError("Provenance collection is too large")
            return _status("omitted", "collection_size_limit", size=len(value))
        items = [
            normalize_value(
                item,
                strict=strict,
                max_encoded_bytes=max_encoded_bytes,
                _depth=_depth + 1,
                _seen=_seen,
            )
            for item in value
        ]
        items.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        normalized = {"type": "set", "values": items}
    else:
        return _unsupported(value, strict=strict, reason="unsupported_type")

    if _encoded_size(normalized) > max_encoded_bytes:
        if strict:
            raise ProvenanceValidationError("Provenance value exceeds the size limit")
        return _status(
            "omitted",
            "encoded_size_limit",
            encoded_bytes=_encoded_size(normalized),
        )
    return normalized
