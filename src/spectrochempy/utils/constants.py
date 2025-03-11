# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = [
    "DEFAULT_DIM_NAME",
    "EPSILON",
    "INPLACE",
    "MASKED",
    "NOMASK",
    "MaskedArray",
    "MaskedConstant",
    "TYPE_BOOL",
    "TYPE_COMPLEX",
    "TYPE_FLOAT",
    "TYPE_INTEGER",
]

# mask constants
import numpy as np
from numpy.ma.core import MaskedArray  # noqa: F401
from numpy.ma.core import MaskedConstant  # noqa: F401
from numpy.ma.core import masked as MASKED  # noqa: F401, N812
from numpy.ma.core import nomask as NOMASK  # noqa: F401, N812

# default dimension names
DEFAULT_DIM_NAME = list("xyzuvwpqrstijklmnoabcdefgh")[::-1]

# epsilon
EPSILON = epsilon = np.finfo(float).eps

# inplace operation
INPLACE = "INPLACE"

# type constants
TYPE_INTEGER = (int, np.int32, np.int64, np.uint32, np.uint64)
TYPE_FLOAT = (float, np.float16, np.float32, np.float64)
TYPE_COMPLEX = (complex, np.complex64, np.complex128)
TYPE_BOOL = (bool, np.bool_)
