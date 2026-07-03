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
# Load a stacked IR dataset and select a single spectral region.

dataset = scp.read_omnic("irdata/nh4y-activation.spg")
region = dataset[:, 2200.0:1800.0]

# %%
# **Normalization** scales each spectrum.  The default is ``method='max'``.

norm = region.normalize(method="max", dim="x")
_ = norm.plot(title="Max-normalized")

# %%
# **Mean-centering** subtracts the mean along a chosen dimension.
# Here we center each spectrum individually (``dim='x'``).

centered = region.center(dim="x")
_ = centered.plot(title="Mean-centered per spectrum")

# %%
# **Autoscaling** mean-centers and divides by the standard deviation.
# This is the classic z-score used before PCA or PLS.

scaled = region.autoscale(dim="x")
_ = scaled.plot(title="Autoscaled (z-score) per spectrum")

# %%
# **Standard Normal Variate (SNV)** is a convenience wrapper that autoscales
# each spectrum individually.  It is equivalent to ``autoscale(dim='x')``.

snv = region.snv()
_ = snv.plot(title="SNV corrected")

# %%
# **Multiplicative Scatter Correction (MSC)** removes multiplicative and
# additive scatter by regressing each spectrum against a mean reference.

msc = region.msc()
_ = msc.plot(title="MSC corrected")

# %%
# This ends the example. Uncomment the next line to display the figures when
# running the script directly with Python.

# scp.show()
