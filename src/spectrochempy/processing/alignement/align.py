# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Compatibility wrapper for the historical ``processing.alignement`` namespace."""

import warnings

from spectrochempy.processing.alignment.align import align
from spectrochempy.processing.alignment.align import can_merge_or_align

__all__ = ["align", "can_merge_or_align"]
__dataset_methods__ = ["align"]

warnings.warn(
    "Importing from `spectrochempy.processing.alignement.align` is deprecated "
    "and will be removed in 0.11.0. Use "
    "`spectrochempy.processing.alignment.align` instead.",
    DeprecationWarning,
    stacklevel=2,
)
