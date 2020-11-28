# -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#

from datetime import datetime
import pickle
import base64

import numpy as np

__all__ = ['json_serialiser', 'json_decoder']

# ======================================================================================================================
# JSON UTILITIES
# ======================================================================================================================

def json_serialiser(byte_obj):
    if isinstance(byte_obj, datetime):
        return {
                "isoformat": byte_obj.isoformat(),
                "__class__": str(byte_obj.__class__)
                }
    if isinstance(byte_obj, np.ndarray):
        # return {"ndarray":byte_obj.tolist(), "dtype": byte_obj.dtype.name}
        return {
                "serialized": base64.b64encode(pickle.dumps(byte_obj)).decode(),
                "__class__" : str(byte_obj.__class__)
                }
    raise ValueError('No encoding handler for data type ' + type(byte_obj))


def json_decoder(dic):
    if "__class__" in dic:
        if dic["__class__"] == str(datetime):
            return datetime.fromisoformat(dic["isoformat"])
        if dic["__class__"] == str(np.ndarray):
            return pickle.loads(base64.b64decode(dic['serialized']))
        raise TypeError("numpy array or datetime expected.")
    return dic


