# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""JSON utilities."""

import base64
import datetime
import json
import pathlib
import pickle
from collections.abc import Mapping

import numpy as np

UNSAFE_LEGACY_LOADING_MESSAGE = (
    "This SCP/PSCP file requires trusted legacy loading because it uses "
    "pickle-based native persistence. Reload with allow_unsafe_legacy=True only "
    "if the file comes from a known and trusted source."
)
SAFE_SCP_DOCUMENT_FORMATS = frozenset({"scp", "pscp"})
SAFE_SCP_DOCUMENT_VERSION = 2


def fromisoformat(s):
    try:
        date = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%Z")
    except Exception:
        date = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    return date


# ======================================================================================
# JSON UTILITIES
# ======================================================================================
def _raise_unsafe_legacy_loading_error():
    from spectrochempy.utils.exceptions import SpectroChemPyError

    raise SpectroChemPyError(UNSAFE_LEGACY_LOADING_MESSAGE)


def _raise_safe_payload_error(message):
    from spectrochempy.utils.exceptions import SpectroChemPyError

    raise SpectroChemPyError(message)


def _validate_safe_document_markers(document):
    has_format = "__format__" in document
    has_version = "__version__" in document

    if not has_format and not has_version:
        return False

    if not has_format or not has_version:
        _raise_safe_payload_error(
            "Malformed SCP/PSCP document: safe-format markers must include both "
            "`__format__` and `__version__`.",
        )

    if document["__format__"] not in SAFE_SCP_DOCUMENT_FORMATS:
        _raise_safe_payload_error(
            f"Unsupported SCP/PSCP document format marker: {document['__format__']!r}.",
        )

    if document["__version__"] != SAFE_SCP_DOCUMENT_VERSION:
        _raise_safe_payload_error(
            "Unsupported SCP/PSCP document version: " f"{document['__version__']!r}.",
        )

    return True


def _is_safe_array_dtype(dtype):
    return not dtype.hasobject


def _decode_safe_array_payload(payload):
    required = {"encoding", "dtype", "shape", "order", "base64"}
    missing = sorted(required - payload.keys())
    if missing:
        _raise_safe_payload_error(
            "Malformed NUMPY_ARRAY payload: missing required field(s) "
            f"{', '.join(missing)}.",
        )

    if payload["encoding"] != "raw-base64":
        _raise_safe_payload_error(
            f"Unsupported NUMPY_ARRAY encoding: {payload['encoding']!r}.",
        )

    try:
        dtype = np.dtype(payload["dtype"])
    except TypeError as exc:
        from spectrochempy.utils.exceptions import SpectroChemPyError

        raise SpectroChemPyError(
            f"Invalid NUMPY_ARRAY dtype: {payload['dtype']!r}.",
        ) from exc

    if not _is_safe_array_dtype(dtype):
        _raise_safe_payload_error(
            "Unsafe NUMPY_ARRAY payload: object dtype is not supported in safe "
            "SCP/PSCP files.",
        )

    shape = payload["shape"]
    if not isinstance(shape, list):
        _raise_safe_payload_error(
            "Malformed NUMPY_ARRAY payload: `shape` must be a JSON list.",
        )
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        _raise_safe_payload_error(
            "Malformed NUMPY_ARRAY payload: `shape` must contain only "
            "non-negative integers.",
        )

    order = payload["order"]
    if order not in {"C", "F"}:
        _raise_safe_payload_error(
            "Malformed NUMPY_ARRAY payload: `order` must be `'C'` or `'F'`.",
        )

    try:
        raw = base64.b64decode(payload["base64"], validate=True)
    except Exception as exc:
        from spectrochempy.utils.exceptions import SpectroChemPyError

        raise SpectroChemPyError(
            "Malformed NUMPY_ARRAY payload: invalid base64 content.",
        ) from exc

    expected_size = dtype.itemsize * int(np.prod(shape, dtype=np.int64))
    if len(raw) != expected_size:
        _raise_safe_payload_error(
            "Malformed NUMPY_ARRAY payload: raw byte length does not match "
            "dtype and shape.",
        )

    try:
        return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape), order=order).copy()
    except ValueError as exc:
        from spectrochempy.utils.exceptions import SpectroChemPyError

        raise SpectroChemPyError(
            "Malformed NUMPY_ARRAY payload: unable to reconstruct ndarray.",
        ) from exc


def _decode_complex_payload(payload):
    required = {"encoding", "dtype", "value"}
    missing = sorted(required - payload.keys())
    if missing:
        _raise_safe_payload_error(
            "Malformed COMPLEX payload: missing required field(s) "
            f"{', '.join(missing)}.",
        )

    if payload["encoding"] != "pair":
        _raise_safe_payload_error(
            f"Unsupported COMPLEX encoding: {payload['encoding']!r}.",
        )

    value = payload["value"]
    if not isinstance(value, list) or len(value) != 2:
        _raise_safe_payload_error(
            "Malformed COMPLEX payload: `value` must be a two-item list "
            "[real, imag].",
        )

    real, imag = value
    if not isinstance(real, int | float) or not isinstance(imag, int | float):
        _raise_safe_payload_error(
            "Malformed COMPLEX payload: real and imaginary parts must be numbers.",
        )

    if payload["dtype"] == "complex":
        return complex(real, imag)

    try:
        dtype = np.dtype(payload["dtype"])
    except TypeError as exc:
        from spectrochempy.utils.exceptions import SpectroChemPyError

        raise SpectroChemPyError(
            f"Invalid COMPLEX dtype: {payload['dtype']!r}.",
        ) from exc

    if dtype.kind != "c":
        _raise_safe_payload_error(
            "Malformed COMPLEX payload: dtype must be a complex dtype.",
        )

    return np.array(complex(real, imag), dtype=dtype)[()]


def _decode_serialized_obj(obj, *, allow_unsafe_legacy=False, safe_document=False):
    from spectrochempy.core.units import Quantity
    from spectrochempy.core.units import Unit
    from spectrochempy.utils.meta import Meta

    if isinstance(obj, list):
        return [
            _decode_serialized_obj(
                item,
                allow_unsafe_legacy=allow_unsafe_legacy,
                safe_document=safe_document,
            )
            for item in obj
        ]

    if not isinstance(obj, Mapping):
        return obj

    decoded = {
        key: _decode_serialized_obj(
            value,
            allow_unsafe_legacy=allow_unsafe_legacy,
            safe_document=safe_document,
        )
        for key, value in obj.items()
    }

    if "__class__" not in decoded:
        return decoded

    klass = decoded["__class__"]
    if klass == "DATETIME":
        return fromisoformat(decoded["isoformat"])
    if klass == "DATETIME64":
        return np.datetime64(decoded["isoformat"])
    if klass == "NUMPY_ARRAY":
        if decoded.get("encoding") == "raw-base64":
            return _decode_safe_array_payload(decoded)
        if "base64" in decoded:
            if safe_document:
                _raise_safe_payload_error(
                    "Malformed safe SCP/PSCP document: legacy pickle-backed "
                    "NUMPY_ARRAY payload encountered.",
                )
            if not allow_unsafe_legacy:
                _raise_unsafe_legacy_loading_error()
            return pickle.loads(base64.b64decode(decoded["base64"]))  # noqa: S301
        if "tolist" in decoded:
            return np.array(decoded["tolist"], dtype=decoded["dtype"])
        _raise_safe_payload_error("Malformed NUMPY_ARRAY payload.")
    if klass == "PATH":
        return pathlib.Path(decoded["str"])
    if klass == "QUANTITY":
        return Quantity.from_tuple(decoded["tuple"])
    if klass == "UNIT":
        return Unit(decoded["str"])
    if klass == "COMPLEX":
        if decoded.get("encoding") == "pair":
            return _decode_complex_payload(decoded)
        if "base64" in decoded:
            if safe_document:
                _raise_safe_payload_error(
                    "Malformed safe SCP/PSCP document: legacy pickle-backed "
                    "COMPLEX payload encountered.",
                )
            if not allow_unsafe_legacy:
                _raise_unsafe_legacy_loading_error()
            return pickle.loads(base64.b64decode(decoded["base64"]))  # noqa: S301
        if "tolist" in decoded:
            if decoded["dtype"] == "complex":
                return complex(decoded["tolist"][0], decoded["tolist"][1])
            return np.array(decoded["tolist"], dtype=decoded["dtype"]).data[()]
        _raise_safe_payload_error("Malformed COMPLEX payload.")
    if klass == "META":
        kwargs = {}
        for key, value in decoded.items():
            if key == "data":
                kwargs.update(value)
        meta = Meta(parent=decoded.get("parent"), name=decoded.get("name"), **kwargs)
        meta.readonly = decoded.get("readonly", False)
        return meta

    raise TypeError(klass)


def json_decoder(dic, allow_unsafe_legacy=False):
    """Decode a serialized json object."""
    return _decode_serialized_obj(
        dic,
        allow_unsafe_legacy=allow_unsafe_legacy,
        safe_document=False,
    )


def json_loads(content, *, allow_unsafe_legacy=False):
    """Load JSON content with explicit legacy unsafe decoding control."""
    raw = json.loads(content)
    safe_document = isinstance(raw, Mapping) and _validate_safe_document_markers(raw)
    return _decode_serialized_obj(
        raw,
        allow_unsafe_legacy=allow_unsafe_legacy,
        safe_document=safe_document,
    )


def json_encoder(byte_obj, encoding=None, *, _root=True):
    """Return a serialised json object."""
    from spectrochempy.application.preferences import PreferencesSet
    from spectrochempy.core.units import Quantity
    from spectrochempy.core.units import Unit

    if byte_obj is None:
        return None

    if hasattr(byte_obj, "_implements"):
        objnames = byte_obj._attributes_()

        dic = {}
        for name in objnames:
            if (
                name in ["readonly"]
                or (name == "dims" and "datasets" in objnames)
                or [name in ["parent", "name"] and isinstance(byte_obj, PreferencesSet)]
                and name not in ["created", "modified", "acquisition_date"]
            ):
                val = getattr(byte_obj, name)
            else:
                val = getattr(byte_obj, f"_{name}")

            # Warning with parent-> circular dependencies!
            if name != "parent":
                if (
                    name == "labels"
                    and isinstance(val, np.ndarray)
                    and val.dtype.hasobject
                ):
                    dic[name] = json_encoder(val.tolist(), encoding=None, _root=False)
                    continue
                dic[name] = json_encoder(val, encoding=encoding, _root=False)
            # we need to differentiate normal dic from Meta object
            if byte_obj._implements("Meta"):
                dic["__class__"] = "META"
        if _root and encoding == "base64":
            implements = byte_obj._implements()
            if implements in {"NDDataset", "Project"}:
                doc_format = "pscp" if implements == "Project" else "scp"
                dic["__format__"] = doc_format
                dic["__version__"] = SAFE_SCP_DOCUMENT_VERSION
        return dic

    if isinstance(byte_obj, str | int | float | bool):
        return byte_obj

    if isinstance(byte_obj, np.bool_):
        return bool(byte_obj)

    if isinstance(byte_obj, np.float64 | np.float32 | float):
        return float(byte_obj)

    if isinstance(byte_obj, np.int64 | np.int32 | int):
        return int(byte_obj)

    if isinstance(byte_obj, tuple):
        return tuple(
            [json_encoder(v, encoding=encoding, _root=False) for v in byte_obj]
        )

    if isinstance(byte_obj, list):
        return [json_encoder(v, encoding=encoding, _root=False) for v in byte_obj]

    if isinstance(byte_obj, dict):
        dic = {}
        for k, v in byte_obj.items():
            dic[k] = json_encoder(v, encoding=encoding, _root=False)
        return dic

    if isinstance(byte_obj, datetime.datetime):
        return {
            "isoformat": byte_obj.strftime("%Y-%m-%dT%H:%M:%S.%f%Z"),
            "__class__": "DATETIME",
        }

    if isinstance(byte_obj, np.datetime64):
        return {
            "isoformat": np.datetime_as_string(byte_obj, timezone="UTC"),
            "__class__": "DATETIME64",
        }

    if isinstance(byte_obj, np.ndarray):
        if encoding is None:
            dtype = byte_obj.dtype
            if str(byte_obj.dtype).startswith("datetime64"):
                byte_obj = np.datetime_as_string(byte_obj, timezone="UTC")
            return {
                "tolist": json_encoder(
                    byte_obj.tolist(), encoding=encoding, _root=False
                ),
                "dtype": str(dtype),
                "__class__": "NUMPY_ARRAY",
            }
        if byte_obj.dtype.hasobject:
            _raise_safe_payload_error(
                "Safe SCP/PSCP writing does not support object-dtype arrays. "
                "Use another format or a trusted legacy workflow.",
            )
        order = (
            "F"
            if byte_obj.ndim > 1
            and byte_obj.flags.f_contiguous
            and not byte_obj.flags.c_contiguous
            else "C"
        )
        safe_array = (
            np.asfortranarray(byte_obj)
            if order == "F"
            else np.ascontiguousarray(byte_obj)
        )
        return {
            "encoding": "raw-base64",
            "dtype": str(byte_obj.dtype),
            "shape": list(byte_obj.shape),
            "order": order,
            "base64": base64.b64encode(safe_array.tobytes(order=order)).decode(),
            "__class__": "NUMPY_ARRAY",
        }

    if isinstance(byte_obj, pathlib.PosixPath | pathlib.WindowsPath):
        return {"str": str(byte_obj), "__class__": "PATH"}

    if isinstance(byte_obj, Unit):
        strunits = f"{byte_obj:D}"
        return {"str": strunits, "__class__": "UNIT"}

    if isinstance(byte_obj, Quantity):
        return {
            "tuple": json_encoder(byte_obj.to_tuple(), encoding=encoding, _root=False),
            "__class__": "QUANTITY",
        }

    if isinstance(byte_obj, np.complex128 | np.complex64 | complex):
        if encoding is None:
            # Handle both NumPy complex and Python's built-in complex
            dtype = str(byte_obj.dtype) if hasattr(byte_obj, "dtype") else "complex"
            return {
                "tolist": json_encoder(
                    [byte_obj.real, byte_obj.imag],
                    encoding=encoding,
                    _root=False,
                ),
                "dtype": dtype,
                "__class__": "COMPLEX",
            }
        return {
            "encoding": "pair",
            "dtype": str(byte_obj.dtype) if hasattr(byte_obj, "dtype") else "complex",
            "value": [float(byte_obj.real), float(byte_obj.imag)],
            "__class__": "COMPLEX",
        }

    raise ValueError(f"No encoding handler for data type {type(byte_obj)}")
