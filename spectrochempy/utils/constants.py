# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Constants
"""

import numpy as np
import quaternion  # noqa: F401
from numpy.ma.core import MaskedArray, MaskedConstant
from numpy.ma.core import masked as MASKED
from numpy.ma.core import nomask as NOMASK

#: Default dimension names.
DEFAULT_DIM_NAME = list("xyzuvwpqrstijklmnoabcdefgh")[::-1]

TYPE_INTEGER = (int, np.int_, np.int32, np.int64, np.uint32, np.uint64)
TYPE_FLOAT = (float, np.float_, np.float32, np.float64)
TYPE_COMPLEX = (complex, np.complex_, np.complex64, np.complex128)
TYPE_BOOL = (bool, np.bool_)
TYPE_QUATERNION = np.dtype(np.quaternion)

#: Minimum value before considering it as zero value.
EPSILON = epsilon = np.finfo(float).eps

#: Flag used to specify inplace slicing.
INPLACE = "INPLACE"
