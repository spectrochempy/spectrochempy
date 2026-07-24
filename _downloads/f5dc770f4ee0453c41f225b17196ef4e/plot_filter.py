# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Savitky-Golay and Whittaker-Eilers smoothing of a Raman spectrum
================================================================
In this example, we use the `Filter` processor to smooth a Raman spectrum.
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
_ = X.plot(color="b", label="original")
ax = Xm.plot(clear=False, color="r", ls="-", lw=1.5, label="SG filter")
diff = X - Xm
s = round(diff.std(dim=-1).values, 2)
ax = diff.plot(clear=False, ls="-", lw=1, label=f"difference (std={s})")
ax.legend(loc="best", fontsize=10)
ax.set_title("Savitzky-Golay filter (size=7, order=2)")

# %%
# Whittaker-Eilers smoothing
# ---------------------------
filter = scp.Filter(method="whittaker", order=2, lamb=1.5)
Xm = filter(X)
_ = X.plot(color="b", label="original")
ax = Xm.plot(clear=False, color="r", ls="-", lw=1.5, label="WE filter")
diff = X - Xm
s = round(diff.std(dim=-1).values, 2)
ax = diff.plot(clear=False, ls="-", lw=1, label=f"difference (std={s})")
ax.legend(loc="best", fontsize=10)
ax.set_title("Whittaker-Eiler filter (order=2, lamb=1.5)")
