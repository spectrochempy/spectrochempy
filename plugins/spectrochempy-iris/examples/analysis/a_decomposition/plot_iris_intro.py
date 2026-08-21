# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
# sphinx_gallery_thumbnail_number = 11

"""
IRIS: 2D-IRIS analysis (plugin)
================================

This example introduces 2D-IRIS analysis of CO adsorption on a sulfide catalyst
with the optional ``spectrochempy-iris`` plugin.

Requires the official ``spectrochempy-iris`` plugin.
Install with: ``pip install spectrochempy[iris]``.
"""

# %%
import spectrochempy as scp

# %%
# Loading and coordinating the dataset
# ------------------------------------
# The example data contain infrared spectra recorded during CO adsorption. The
# dataset has wavenumber coordinates along ``x`` and acquisition timestamps along
# ``y``.

X = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")
X.coordset

# %%
# The IRIS model is easier to interpret with pressure coordinates along the
# observation axis, so we attach the measured CO pressures to ``y``.

pressures = [
    0.003,
    0.004,
    0.009,
    0.014,
    0.021,
    0.026,
    0.036,
    0.051,
    0.093,
    0.150,
    0.203,
    0.300,
    0.404,
    0.503,
    0.602,
    0.702,
    0.801,
    0.905,
    1.004,
]
c_pressures = scp.Coord(pressures, title="pressure", units="torr")

# %%
# Keep the original time coordinate as a secondary coordinate, and make pressure
# the active one for plotting and IRIS fitting.

c_times = X.y.copy()
X.y = [c_times, c_pressures]
X.y.select(2)
X.coordset

# %%
# We now select the CO adsorption spectral region.

X_ = X[:, 2250.0:1950.0]
_ = X_.plot(colorbar=True)
_ = X_.plot_contourf(colorbar=True)

# %%
# IRIS analysis without regularization
# ------------------------------------
# The plugin exposes its workflows through ``scp.iris`` and also adds
# dataset-bound helpers under ``dataset.iris``. We start by building the
# Langmuir kernel from the dataset accessor.

K = X_.iris.kernel_matrix(kernel_type="langmuir", q=[-8, -1, 50])
K.kernel

# %%
# The model can then be fitted with no explicit regularization.

iris1 = scp.iris.IRIS(log_level="INFO")
_ = iris1.fit(X_, K)

# %%
# Grouped fitted outputs are available from ``result``. Historical direct
# attributes such as ``f`` remain supported, but the grouped result is the
# preferred way to inspect the fit.

f = iris1.result.f
_ = iris1.result.K

_ = f.plot_contour()
_ = iris1.plotmerit()

# %%
# Manual regularization search
# ----------------------------
# A second fit scans a regularization range manually and displays the L-curve.

iris2 = scp.iris.IRIS(reg_par=[-10, 1, 12])
_ = iris2.fit(X_, K)
_ = iris2.plotlcurve(title="L curve, manual search")

# %%
# The best regularization parameter is visually near index ``-6`` for this
# dataset, corresponding to a lambda around ``1e-4``.

_ = iris2.result.f[-6].plot_contour()
_ = iris2.plotmerit(-6)

# %%
# Automatic search
# ----------------
# We can then refine the search around the visually selected range.

iris3 = scp.iris.IRIS(log_level="INFO", reg_par=[-6, -2])
_ = iris3.fit(X_, K)
_ = iris3.plotlcurve(title="L curve, automated search")

# %%
# The largest curvature of the L-curve is at index 5 for this automated search.

_ = iris3.result.f[5].plot_contour()
_ = iris3.plotmerit(5)

# %%
# The historical root example also demonstrated the legacy direct classes
# ``IRIS`` and ``IrisKernel``. New gallery material should prefer the plugin
# namespace shown above: ``scp.iris.IRIS`` and ``dataset.iris.kernel_matrix``.


# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.
#
# # scp.show()
