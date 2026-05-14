# ======================================================================================
# Copyright © 2014–2026
# Laboratoire Catalyse et Spectrochimie (LCS),
# Université de Caen Normandie
# SPDX-License-Identifier: CECILL-B
# ======================================================================================

"""
Matplotlib assets (stylesheets only) management.

DEPRECATED: This module is now a re-export wrapper.
The implementation has moved to spectrochempy.plotting._mpl_assets.

Note: Font installation has been removed. SpectroChemPy now relies on
Matplotlib's built-in fonts.
"""

from spectrochempy.plotting._mpl_assets import ensure_mpl_assets_installed

ensure_mpl_assets_installed = ensure_mpl_assets_installed
