# ======================================================================================
# Copyright © 2014–2026
# Laboratoire Catalyse et Spectrochimie (LCS),
# Université de Caen Normandie
# SPDX-License-Identifier: CECILL-B
# ======================================================================================

"""
Silent installation of SpectroChemPy Matplotlib assets (stylesheets only).

Note: Font installation has been removed. SpectroChemPy now relies on
Matplotlib's built-in fonts (DejaVu Sans, DejaVu Serif, etc.) for all
rendering. This simplifies the codebase and eliminates font-related warnings.

Users who need specific fonts should install them system-wide or use
text.usetex=True with a LaTeX installation.
"""

from pathlib import Path

from spectrochempy.utils._logging import debug_


def ensure_mpl_assets_installed():
    """
    Ensure SpectroChemPy Matplotlib stylesheets are installed.

    Note: Font installation has been removed. Only stylesheets are managed.

    Failures are reported via warning_(), but never raise exceptions.
    """
    try:
        _install_stylesheets()
    except Exception as exc:  # defensive: never crash plotting init
        from spectrochempy.utils._logging import warning_

        warning_(
            "Failed to install SpectroChemPy matplotlib stylesheets "
            f"({exc.__class__.__name__}: {exc})"
        )


def _install_stylesheets():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    stylesheets_src = Path(__file__).parent / "stylesheets"
    if not stylesheets_src.exists():
        debug_("No SpectroChemPy matplotlib stylesheets directory found")
        return

    style_files = list(stylesheets_src.glob("*.mplstyle"))
    if not style_files:
        debug_("No SpectroChemPy stylesheets to install")
        return

    # User stylelib (used by plt.style.use)
    user_stylelib = Path(mpl.get_configdir()) / "stylelib"
    user_stylelib.mkdir(parents=True, exist_ok=True)

    # System stylelib (used by SpectroChemPy preferences)
    system_stylelib = Path(mpl.get_data_path()) / "stylelib"
    system_stylelib.mkdir(parents=True, exist_ok=True)

    installed = False

    for src in style_files:
        for dest_dir in (user_stylelib, system_stylelib):
            dest = dest_dir / src.name
            if not dest.exists():
                import shutil

                shutil.copy(src, dest)
                debug_("Installed stylesheet %s → %s", src.name, dest_dir)
                installed = True

    if installed:
        plt.style.reload_library()
    else:
        debug_("SpectroChemPy matplotlib stylesheets already installed")
