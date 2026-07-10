# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Chemometric preprocessing
=========================

This example demonstrates the standard preprocessing operations
available in SpectroChemPy: normalization, mean-centering, autoscaling,
Standard Normal Variate (SNV), and Multiplicative Scatter Correction (MSC).
"""

import spectrochempy as scp

# %%
# Load the dataset
# -----------------
dataset = scp.read_omnic("irdata/nh4y-activation.spg")
region = dataset[:, 4000.0:2000.0]

# %%
# Normalization
# --------------
# Scales each spectrum.  The default is ``method='max'``:
norm = region.normalize(method="max", dim="x")
_ = norm.plot(title="Max-normalized")

# %%
# Mean-centering
# ---------------
# Subtracts the mean along a chosen dimension:
centered = region.center(dim="x")
_ = centered.plot(title="Mean-centered per spectrum")

# %%
# Autoscaling
# ------------
# Mean-centers and divides by the standard deviation (z-score):
scaled = region.autoscale(dim="x")
_ = scaled.plot(title="Autoscaled (z-score) per spectrum")

# %%
# SNV and MSC
# ------------
# Standard Normal Variate (SNV) — equivalent to ``autoscale(dim='x')``:
snv = region.snv()
_ = snv.plot(title="SNV corrected")

# %%
# Multiplicative Scatter Correction (MSC):
msc = region.msc()
_ = msc.plot(title="MSC corrected")

# %%
# This ends the example. Uncomment the next line to display the figures when
# running the script directly with Python.

# scp.show()
