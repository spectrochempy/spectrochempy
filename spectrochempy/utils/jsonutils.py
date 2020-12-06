# -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#
"""JSON utilities

"""
from datetime import datetime
import pickle
import base64
import pathlib
import numpy as np

from spectrochempy.units import Quantity, Unit


__all__ = ['json_serialiser', 'json_decoder']


# ======================================================================================================================
# JSON UTILITIES
# ======================================================================================================================

def json_serialiser__(byte_obj):
    """
    Return a serialised object
    """
    if isinstance(byte_obj, datetime):
        return {
                "isoformat": byte_obj.isoformat(),
                "__class__": str(byte_obj.__class__)
                }
    elif isinstance(byte_obj, np.ndarray):
        # return {"ndarray":byte_obj.tolist(), "dtype": byte_obj.dtype.name}
        return {
                "serialized": base64.b64encode(pickle.dumps(byte_obj)).decode(),
                "__class__": str(byte_obj.__class__)
                }
    elif isinstance(byte_obj, pathlib.PosixPath):
        return {
                "str": str(byte_obj),
                "__class__": str(byte_obj.__class__)
                }

    elif isinstance(byte_obj, Quantity):
        return {
                "tuple": byte_obj.to_tuple(),
                "__class__": str(byte_obj.__class__)
                }
    raise ValueError(f'No encoding handler for data type {type(byte_obj)}')


def json_decoder(dic):
    """Decode a serialised object
    """
    if "__class__" in dic:
        if dic["__class__"] == str(datetime):
            return datetime.fromisoformat(dic["isoformat"])
        elif dic["__class__"] == str(np.ndarray):
            return pickle.loads(base64.b64decode(dic['serialized']))
        elif dic["__class__"] == str(pathlib.PosixPath):
            return pathlib.Path(dic["str"])
        elif dic["__class__"] == str(Quantity):
            return Quantity.from_tuple(dic["tuple"])
        raise TypeError("numpy array, quantity, datetime or pathlib.PosixPath")
    return dic


def json_serialiser(byte_obj):
    """
    Return a serialised object
    """
    from spectrochempy.utils import Meta

    if hasattr(byte_obj, 'implements'):
        objnames = dir(byte_obj)
        dic = {}
        for names in objnames:
            val = getattr(byte_obj, f'_{names}')
            if val is not None:
                dic[names] = json_serialiser(val)
        return dic

    elif isinstance(byte_obj, (str, int, float, complex)):
        return byte_obj

    elif isinstance(byte_obj, (tuple, list)):
        return [json_serialiser(val) for val in byte_obj]

    elif isinstance(byte_obj, (Meta, dict)):
        return {k: json_serialiser(val) for k, val in byte_obj.items()}

    elif isinstance(byte_obj, datetime):
        return {
                "isoformat": byte_obj.isoformat(),
                "__class__": str(byte_obj.__class__)
                }
    elif isinstance(byte_obj, np.ndarray):
        # return {"ndarray":byte_obj.tolist(), "dtype": byte_obj.dtype.name}
        return {
                "serialized": base64.b64encode(pickle.dumps(byte_obj)).decode(),
                "__class__": str(byte_obj.__class__)
                }
    elif isinstance(byte_obj, pathlib.PosixPath):
        return {
                "str": str(byte_obj),
                "__class__": str(byte_obj.__class__)
                }

    elif isinstance(byte_obj, Unit):
        return {
                "str": str(byte_obj),
                "__class__": str(byte_obj.__class__)
                }

    elif isinstance(byte_obj, Quantity):
        return {
                "tuple": byte_obj.to_tuple(),
                "__class__": str(byte_obj.__class__)
                }
    raise ValueError(f'No encoding handler for data type {type(byte_obj)}')


if __name__ == '__main__':
    import json
    from spectrochempy import read_omnic

    nd = read_omnic('wodger.spg')

    js = json_serialiser(nd)


    js_string = json.dumps(nd, default=json_serialiser, indent=2)
