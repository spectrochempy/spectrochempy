# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""JSON utilities."""

import base64
import datetime
import pathlib
import pickle

import numpy as np


def fromisoformat(s):
    try:
        date = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%Z")
    except Exception:
        date = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    return date


# ======================================================================================
# JSON UTILITIES
# ======================================================================================
def json_decoder(dic):
    """Decode a serialized json object."""
    from spectrochempy.core.dataset.baseobjects.meta import Meta
    from spectrochempy.core.units import Quantity
    from spectrochempy.core.units import Unit

    if "__class__" in dic:
        klass = dic.pop("__class__")
        if klass == "DATETIME":
            return fromisoformat(dic["isoformat"])
        if klass == "DATETIME64":
            return np.datetime64(dic["isoformat"])
        if klass == "NUMPY_ARRAY":
            if "base64" in dic:
                return pickle.loads(base64.b64decode(dic["base64"]))  # noqa: S301
            if "tolist" in dic:
                return np.array(dic["tolist"], dtype=dic["dtype"])
        elif klass == "PATH":
            return pathlib.Path(dic["str"])
        elif klass == "QUANTITY":
            return Quantity.from_tuple(dic["tuple"])
        elif klass == "UNIT":
            return Unit(dic["str"])
        elif klass == "COMPLEX":
            if "base64" in dic:
                return pickle.loads(base64.b64decode(dic["base64"]))  # noqa: S301
            if "tolist" in dic:
                return np.array(dic["tolist"], dtype=dic["dtype"]).data[()]
        elif klass == "META":
            kwargs = {}
            for k, v in dic.items():
                if k == "data":
                    for kk, vv in v.items():
                        kwargs[kk] = json_decoder(vv) if isinstance(vv, dict) else vv
            meta = Meta(parent=dic.get("parent"), name=dic.get("name"), **kwargs)
            meta.readonly = dic.get("readonly", False)
            return meta

        raise TypeError(dic["__class__"])

    return dic


def json_encoder(byte_obj, encoding=None):
    """Return a serialised json object."""
    from spectrochempy.core.dataset.arraymixins.ndplot import PreferencesSet
    from spectrochempy.core.units import Quantity
    from spectrochempy.core.units import Unit

    if byte_obj is None:
        return None

    if hasattr(byte_obj, "_implements"):
        objnames = byte_obj.__dir__()

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
                dic[name] = json_encoder(val, encoding=encoding)
            # we need to differentiate normal dic from Meta object
            if byte_obj._implements("Meta"):
                dic["__class__"] = "META"
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
        return tuple([json_encoder(v, encoding=encoding) for v in byte_obj])

    if isinstance(byte_obj, list):
        return [json_encoder(v, encoding=encoding) for v in byte_obj]

    if isinstance(byte_obj, dict):
        dic = {}
        for k, v in byte_obj.items():
            dic[k] = json_encoder(v, encoding=encoding)
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
                "tolist": json_encoder(byte_obj.tolist(), encoding=encoding),
                "dtype": str(dtype),
                "__class__": "NUMPY_ARRAY",
            }
        return {
            "base64": base64.b64encode(pickle.dumps(byte_obj)).decode(),
            "__class__": "NUMPY_ARRAY",
        }

    if isinstance(byte_obj, pathlib.PosixPath | pathlib.WindowsPath):
        return {"str": str(byte_obj), "__class__": "PATH"}

    if isinstance(byte_obj, Unit):
        strunits = f"{byte_obj:D}"
        return {"str": strunits, "__class__": "UNIT"}

    if isinstance(byte_obj, Quantity):
        return {
            "tuple": json_encoder(byte_obj.to_tuple(), encoding=encoding),
            "__class__": "QUANTITY",
        }

    if isinstance(byte_obj, np.complex128 | np.complex64 | complex):
        if encoding is None:
            return {
                "tolist": json_encoder(
                    [byte_obj.real, byte_obj.imag],
                    encoding=encoding,
                ),
                "dtype": str(byte_obj.dtype),
                "__class__": "COMPLEX",
            }
        return {
            "base64": base64.b64encode(pickle.dumps(byte_obj)).decode(),
            "__class__": "COMPLEX",
        }

    raise ValueError(f"No encoding handler for data type {type(byte_obj)}")
