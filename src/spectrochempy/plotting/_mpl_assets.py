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
from spectrochempy.utils._logging import warning_


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


# Generated stylesheets shipped by mistake in earlier releases, by SHA-256.
# Installed copies are removed only when they are unmodified.
_OBSOLETE_STYLESHEETS = {
    # spectrochempy 1.1.1 to 1.1.3
    "mycustom.mplstyle": (
        "9a6c66e8ee4fda8ac86a9e2e6c4fc385c86c39c1cad9be3a85d75e455b67971f"
    ),
}


def _remove_obsolete_stylesheets(*stylelibs):
    import hashlib

    removed = False
    for stylelib in stylelibs:
        for name, digest in _OBSOLETE_STYLESHEETS.items():
            path = stylelib / name
            try:
                if (
                    path.is_file()
                    and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                ):
                    path.unlink()
                    debug_("Removed obsolete stylesheet %s", path)
                    removed = True
            except Exception as exc:
                warning_(f"Failed to remove obsolete stylesheet {path}: {exc}")
    return removed


def _install_stylesheets():
    import matplotlib as mpl

    # Done before importing pyplot, which parses the user stylelib.
    removed = _remove_obsolete_stylesheets(
        Path(mpl.get_configdir()) / "stylelib",
        Path(mpl.get_data_path()) / "stylelib",
    )

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
    try:
        user_stylelib.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        warning_(f"Failed to create user stylelib directory: {exc}")

    # System stylelib (used by SpectroChemPy preferences)
    system_stylelib = Path(mpl.get_data_path()) / "stylelib"
    try:
        system_stylelib.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        warning_(f"Failed to create system stylelib directory: {exc}")

    installed = removed

    for src in style_files:
        # Try user stylelib first
        try:
            dest = user_stylelib / src.name
            if not dest.exists():
                import shutil

                shutil.copy(src, dest)
                debug_("Installed stylesheet %s → %s", src.name, user_stylelib)
                installed = True
        except Exception as exc:
            warning_(f"Failed to install stylesheet to user directory: {exc}")

        # Try system stylelib separately
        try:
            dest = system_stylelib / src.name
            if not dest.exists():
                import shutil

                shutil.copy(src, dest)
                debug_("Installed stylesheet %s → %s", src.name, system_stylelib)
                installed = True
        except Exception as exc:
            warning_(f"Failed to install stylesheet to system directory: {exc}")

    if installed:
        plt.style.reload_library()
    else:
        debug_("SpectroChemPy matplotlib stylesheets already installed")
