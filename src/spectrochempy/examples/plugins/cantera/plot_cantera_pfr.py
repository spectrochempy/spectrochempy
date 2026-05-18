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
the :class:`~spectrochempy.PFR` plug flow reactor simulation class.
"""

# %%
import spectrochempy as scp

from importlib.util import find_spec

if find_spec("spectrochempy_cantera") is None:
    raise ImportError(
        "This example requires the optional spectrochempy-cantera plugin.\n"
        "Install it with: pip install spectrochempy[cantera]"
    )

# %%
# When the plugin is installed, :class:`~spectrochempy.PFR` is available
# in the ``scp`` namespace:

print(f"PFR class: {scp.PFR}")

# %%
# The PFR constructor expects a Cantera mechanism (CTI/XML mechanism file).
# See the :class:`~spectrochempy.PFR` API reference for full parameter details.

# %%
# If the plugin is not installed, accessing ``scp.PFR`` raises a
# :class:`~spectrochempy.plugins.deps.MissingPluginError`.

# scp.show()
