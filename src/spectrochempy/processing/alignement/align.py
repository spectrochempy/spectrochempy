# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Compatibility wrapper for the historical ``processing.alignement`` namespace."""

from spectrochempy.processing.alignment.align import align
from spectrochempy.processing.alignment.align import can_merge_or_align
from spectrochempy.utils.decorators import warn_deprecated

__all__ = ["align", "can_merge_or_align"]
__dataset_methods__ = ["align"]

warn_deprecated(
    "spectrochempy.processing.alignement.align",
    kind="import path",
    replace="spectrochempy.processing.alignment.align",
    policy=True,
    action="is deprecated",
    stacklevel=2,
)
