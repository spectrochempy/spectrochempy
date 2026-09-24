# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Public supervised cross-validation interface."""

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

from spectrochempy.analysis._cross_validation import CrossValidationResult
from spectrochempy.analysis._cross_validation import cross_validate

__all__ = [
    "CrossValidationResult",
    "GroupKFold",
    "KFold",
    "LeaveOneOut",
    "cross_validate",
]
