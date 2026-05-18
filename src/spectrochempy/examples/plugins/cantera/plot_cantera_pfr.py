# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Cantera: PFR reactor simulation (plugin)
=========================================

This example shows how the optional ``spectrochempy-cantera`` plugin provides
the ``scp.cantera.PFR`` plug flow reactor simulation callable.
"""

# %%
import spectrochempy as scp

from importlib.util import find_spec

OPTIONAL_PLUGIN = "spectrochempy_cantera"

if find_spec(OPTIONAL_PLUGIN) is None:
    print(
        "This example requires the optional spectrochempy-cantera plugin.\n"
        "Install it with: pip install spectrochempy[cantera]"
    )
else:
    # %%
    # When the plugin is installed, ``PFR`` is available from the ``scp.cantera``
    # plugin namespace. It is a lazy callable that imports the implementation
    # only when used:

    print(f"PFR callable: {scp.cantera.PFR}")

    # %%
    # The PFR constructor currently expects inputs compatible with the legacy
    # Cantera mechanism API. Full construction examples will be added after the
    # PFR implementation is adapted to Cantera 3.2+.

# %%
# If the plugin is not installed, accessing ``scp.cantera`` raises a clear
# installation hint for ``spectrochempy-cantera``.

# scp.show()
