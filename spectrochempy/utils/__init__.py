# -*- coding: utf-8 -*-
# flake8: noqa

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Package containing various utilities classes and functions.
"""
# some useful constants
# ------------------------------------------------------------------
# import numpy as np

# masked arrays
# ------------------------------------------------------------------
from numpy.ma.core import (
    masked as MASKED,
    nomask as NOMASK,
    MaskedArray,
    MaskedConstant,
)  # noqa: F401

# import util files content
# ------------------------------------------------------------------

from .print import *
from .file import *
from .jsonutils import *
from .misc import *
from .packages import *
from .plots import *
from .system import *
from .traits import *
from .zip import *
from .exceptions import *
from .version import *
