# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
NMR: reading TopSpin files (plugin)
=====================================

This example shows how to read Bruker TopSpin NMR files using the
optional ``spectrochempy-nmr`` plugin.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
import spectrochempy as scp

# %%
# ``read_topspin`` is registered under the ``scp.nmr`` plugin namespace.
# The top-level ``scp.read_topspin`` alias is kept for compatibility.

ds = scp.nmr.read_topspin(
    scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d",
    expno=1,
    remove_digital_filter=True,
)

print(f"Loaded dataset: {ds}")
print(f"Shape: {ds.shape}")

# %%
# Plot the spectrum:

_ = ds.plot()

# %%
# If the plugin is not installed, the function or method raises a
# :class:`~spectrochempy.plugins.deps.MissingPluginError` with installation
# instructions.

# scp.show()
