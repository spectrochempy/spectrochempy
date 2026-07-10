# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Compare smoothing window sizes on a Raman spectrum
=================================================

This example shows how the smoothing window size changes the balance between
noise reduction and signal distortion on a Raman spectrum.
"""

import spectrochempy as scp

# %%
# Prepare a noisy spectrum
# -------------------------
dataset = scp.read("ramandata/labspec/SMC1-Initial_RT.txt")
region = dataset[:, :400.0]
region += 200 * scp.random(region.shape)
region.name = "noisy spectrum"

prefs = scp.preferences
prefs.figure.figsize = (8, 4)
_ = region.plot()

# %%
# Moving-average smoothing
# -------------------------
smoothed = {size: region.smooth(size) for size in (3, 7, 11)}

# %%
# Compare the results. Larger windows remove more noise but can also flatten
# narrow features:
for size, smoothed_region in smoothed.items():
    _ = scp.plot_compare(
        region,
        smoothed_region,
        title=f"Moving-average smoothing (size={size})",
    )

# %%
# Savitzky-Golay smoothing
# -------------------------
sg = scp.Filter(method="savgol", size=7, order=2)(region)
_ = scp.plot_compare(region, sg, title="Savitzky-Golay smoothing (size=7, order=2)")

# %%
# This ends the example. Uncomment the next line to display the figures when
# running the script directly with Python.

# scp.show()
