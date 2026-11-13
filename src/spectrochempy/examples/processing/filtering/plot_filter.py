# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Savitzky-Golay and Whittaker-Eilers smoothing of a Raman spectrum
=================================================================
In this example, we use the ``Filter`` processor to smooth a Raman spectrum.
"""

import spectrochempy as scp

# %%
# Load and plot the spectrum
# ---------------------------
X = scp.read("ramandata/labspec/SMC1-Initial_RT.txt")
prefs = scp.preferences
prefs.figure.figsize = (8, 4)
_ = X.plot()

# %%
# Savitzky-Golay filtering
# -------------------------
filter = scp.Filter(method="savgol", size=5, order=0)

Xm = filter(X)
diff = X - Xm
s = round(diff.std(dim=-1).values, 2)
_ = scp.plot_multiple(
    [X, Xm, diff],
    method="pen",
    labels=["original", "SG filter", f"difference (std={s})"],
    color=["b", "r", "C2"],
    ls="-",
    lw=[1, 1.5, 1],
    legend="best",
    title="Savitzky-Golay filter (size=5, order=0)",
)

# %%
# Whittaker-Eilers smoothing
# ---------------------------
filter = scp.Filter(method="whittaker", order=2, lamb=1.5)
Xm = filter(X)
diff = X - Xm
s = round(diff.std(dim=-1).values, 2)
_ = scp.plot_multiple(
    [X, Xm, diff],
    method="pen",
    labels=["original", "WE filter", f"difference (std={s})"],
    color=["b", "r", "C2"],
    ls="-",
    lw=[1, 1.5, 1],
    legend="best",
    title="Whittaker-Eilers filter (order=2, lamb=1.5)",
)
