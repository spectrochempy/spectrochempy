# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# ruff: noqa

__all__ = [
    "_base",
    "crossdecomposition",
    "curvefitting",
    "decomposition",
    "integration",
    "kinetic",
    "peakfinding",
]

from . import _base
from . import crossdecomposition
from . import curvefitting
from . import decomposition
from . import integration
from . import kinetic
from . import peakfinding
