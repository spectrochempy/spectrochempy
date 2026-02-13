# ======================================================================================
# Copyright © 2014-2026
# Laboratoire Catalyse et Spectrochimie (LCS),
# Université de Caen Normandie
# SPDX-License-Identifier: CECILL-B
# ======================================================================================

"""
Matplotlib assets (stylesheets and fonts) management.

DEPRECATED: This module is now a re-export wrapper.
The implementation has moved to spectrochempy.plotting._mpl_assets.
"""

# Re-export from new location for backward compatibility
from spectrochempy.plotting._mpl_assets import (
    ensure_mpl_assets_installed,
    get_installed_style_dirs,
)

ensure_mpl_assets_installed = ensure_mpl_assets_installed
get_installed_style_dirs = get_installed_style_dirs
