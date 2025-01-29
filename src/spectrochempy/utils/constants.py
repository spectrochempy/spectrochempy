# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np
from numpy.ma.core import MaskedArray  # noqa: F401
from numpy.ma.core import MaskedConstant  # noqa: F401
from numpy.ma.core import masked as MASKED  # noqa: F401, N812
from numpy.ma.core import nomask as NOMASK  # noqa: F401, N812

#: Default dimension names.
DEFAULT_DIM_NAME = list("xyzuvwpqrstijklmnoabcdefgh")[::-1]

#: Minimum value before considering it as zero value.
EPSILON = epsilon = np.finfo(float).eps

#: Flag used to specify inplace slicing.
INPLACE = "INPLACE"
