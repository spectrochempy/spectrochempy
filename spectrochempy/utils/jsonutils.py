# -*- coding: utf-8 -*-

#  =====================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =
#  =====================================================================================================================
#
"""
JSON utilities.
"""
from datetime import datetime
import pickle
import base64
import pathlib

import numpy as np

from spectrochempy.units import Quantity, Unit

__all__ = ["json_serialiser", "json_decoder"]


def fromisoformat(s):
    try:
        date = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%Z")
    except Exception:
        date = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    return date


# ======================================================================================================================
# JSON UTILITIES
# ======================================================================================================================


def json_decoder(dic):
    """
    Decode a serialised json object.
    """

    if "__class__" in dic:

        klass = dic["__class__"]
        if klass == "DATETIME":
            return fromisoformat(dic["isoformat"])
        elif klass == "NUMPY_ARRAY":
            if "base64" in dic:
                return pickle.loads(base64.b64decode(dic["base64"]))
            elif "tolist" in dic:
                return np.array(dic["tolist"], dtype=dic["dtype"])
        elif klass == "PATH":
            return pathlib.Path(dic["str"])
        elif klass == "QUANTITY":
            return Quantity.from_tuple(dic["tuple"])
        elif klass == "UNIT":
            return Unit(dic["str"])
        elif klass == "COMPLEX":
            if "base64" in dic:
                return pickle.loads(base64.b64decode(dic["base64"]))
            elif "tolist" in dic:
                return np.array(dic["tolist"], dtype=dic["dtype"]).data[()]

        raise TypeError(dic["__class__"])

    return dic


def json_serialiser(byte_obj, encoding=None):
    """
    Return a serialised json object.
    """
    from spectrochempy.core.dataset.ndplot import PreferencesSet

    if byte_obj is None:
        return None

    elif hasattr(byte_obj, "implements"):

        objnames = byte_obj.__dir__()
        # particular case of Linear Coordinates
        if byte_obj.implements("LinearCoord"):
            objnames.remove("data")
        dic = {}
        for name in objnames:

            if (
                name in ["readonly"]
                or (name == "dims" and "datasets" in objnames)
                or [name in ["parent", "name"] and isinstance(byte_obj, PreferencesSet)]
            ):
                val = getattr(byte_obj, name)
            else:
                val = getattr(byte_obj, f"_{name}")

            # Warning with parent-> circular dependencies!
            if name != "parent":
                dic[name] = json_serialiser(val, encoding=encoding)
        return dic

    elif isinstance(byte_obj, (str, int, float, bool)):
        return byte_obj

    elif isinstance(byte_obj, np.bool_):
        return bool(byte_obj)

    elif isinstance(byte_obj, (np.float64, np.float32, float)):
        return float(byte_obj)

    elif isinstance(byte_obj, (np.int64, np.int32, int)):
        return int(byte_obj)

    elif isinstance(byte_obj, tuple):
        return tuple([json_serialiser(v, encoding=encoding) for v in byte_obj])

    elif isinstance(byte_obj, list):
        return [json_serialiser(v, encoding=encoding) for v in byte_obj]

    elif isinstance(byte_obj, dict):
        dic = {}
        for k, v in byte_obj.items():
            dic[k] = json_serialiser(v, encoding=encoding)
        return dic

    elif isinstance(byte_obj, datetime):
        return {
            "isoformat": byte_obj.strftime("%Y-%m-%dT%H:%M:%S.%f%Z"),
            "__class__": "DATETIME",
        }  # .isoformat()

    elif isinstance(byte_obj, np.ndarray):
        if encoding is None:
            return {
                "tolist": json_serialiser(byte_obj.tolist(), encoding=encoding),
                "dtype": str(byte_obj.dtype),
                "__class__": "NUMPY_ARRAY",
            }
        else:
            return {
                "base64": base64.b64encode(pickle.dumps(byte_obj)).decode(),
                "__class__": "NUMPY_ARRAY",
            }

    elif isinstance(byte_obj, pathlib.PosixPath):
        return {"str": str(byte_obj), "__class__": "PATH"}

    elif isinstance(byte_obj, Unit):
        return {"str": str(byte_obj), "__class__": "UNIT"}

    elif isinstance(byte_obj, Quantity):
        return {
            "tuple": json_serialiser(byte_obj.to_tuple(), encoding=encoding),
            "__class__": "QUANTITY",
        }

    elif isinstance(byte_obj, (np.complex128, np.complex64, np.complex)):
        if encoding is None:
            return {
                "tolist": json_serialiser(
                    [byte_obj.real, byte_obj.imag], encoding=encoding
                ),
                "dtype": str(byte_obj.dtype),
                "__class__": "COMPLEX",
            }
        else:
            return {
                "base64": base64.b64encode(pickle.dumps(byte_obj)).decode(),
                "__class__": "COMPLEX",
            }

    raise ValueError(f"No encoding handler for data type {type(byte_obj)}")


if __name__ == "__main__":
    pass
