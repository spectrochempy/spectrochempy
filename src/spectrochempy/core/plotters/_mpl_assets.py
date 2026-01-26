# ======================================================================================
# Copyright © 2014–2026
# Laboratoire Catalyse et Spectrochimie (LCS),
# Université de Caen Normandie
# SPDX-License-Identifier: CECILL-B
# ======================================================================================

"""
Silent installation of SpectroChemPy Matplotlib assets (stylesheets and fonts).

This module:
- Performs filesystem operations only if needed
- Is idempotent
- Never raises on failure (best-effort)
- Reports issues via SpectroChemPy logging helpers
"""

import shutil
from pathlib import Path

from spectrochempy.application.application import debug_
from spectrochempy.application.application import warning_


def ensure_mpl_assets_installed():
    """
    Ensure SpectroChemPy Matplotlib stylesheets and fonts are installed.

    Failures are reported via warning_(), but never raise exceptions.
    """
    try:
        _install_stylesheets()
        _install_fonts()
    except Exception as exc:  # defensive: never crash plotting init
        warning_(
            "Failed to install SpectroChemPy matplotlib assets "
            f"({exc.__class__.__name__}: {exc})"
        )


def _install_stylesheets():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    stylesheets_src = Path(__file__).parent / "stylesheets"
    if not stylesheets_src.exists():
        debug_("No SpectroChemPy matplotlib stylesheets directory found")
        return

    cfgdir = Path(mpl.get_configdir())
    stylelib = cfgdir / "stylelib"
    stylelib.mkdir(parents=True, exist_ok=True)

    styles = list(stylesheets_src.glob("*.mplstyle"))
    missing = [s for s in styles if not (stylelib / s.name).exists()]

    if not missing:
        debug_("SpectroChemPy matplotlib stylesheets already installed")
        return

    debug_("Installing %d matplotlib stylesheet(s)", len(missing))
    for src in missing:
        shutil.copy(src, stylelib / src.name)
        debug_("Installed stylesheet: %s", src.name)

    plt.style.reload_library()


def _install_fonts():
    import matplotlib as mpl
    from matplotlib import get_cachedir

    fonts_src = Path(__file__).parent / "fonts"
    if not fonts_src.exists():
        debug_("No SpectroChemPy fonts directory found")
        return

    fonts_dest = Path(mpl.get_data_path()) / "fonts" / "ttf"
    fonts_dest.mkdir(parents=True, exist_ok=True)

    fonts = list(fonts_src.glob("*.[ot]tf"))
    missing = [f for f in fonts if not (fonts_dest / f.name).exists()]

    if not missing:
        debug_("SpectroChemPy fonts already installed")
        return

    debug_("Installing %d matplotlib font(s)", len(missing))
    for src in missing:
        shutil.copy(src, fonts_dest / src.name)
        debug_("Installed font: %s", src.name)

    # Clear font cache only if new fonts were installed
    cache_dir = Path(get_cachedir())
    for cache_file in cache_dir.glob("*.cache"):
        if cache_file.is_file():
            cache_file.unlink()
            debug_("Cleared font cache: %s", cache_file.name)
