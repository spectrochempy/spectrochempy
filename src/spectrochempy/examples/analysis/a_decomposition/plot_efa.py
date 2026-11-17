# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Follow the evolution of factors in an IR series
===============================================

Use evolving factor analysis to follow the first five eigenvalue curves
as successive IR spectra are added.
See the :doc:`EFA tutorial </userguide/analysis/efa>` for the backward
curves and how to interpret the two directions together.
"""

import spectrochempy as scp

spectra = scp.read_omnic("irdata/nh4y-activation.spg")[::3]
spectra.y -= spectra.y[0]
spectra.y.ito("min")
spectra.y.title = "elapsed time"
spectra[:, 1230.0:920.0] = scp.MASKED
spectra[:, 5997.0:5993.0] = scp.MASKED

# %%
# Inspect every third spectrum from the series used in the fit. The gaps
# mark the masked regions.

_ = spectra[::3].plot(method="lines", title="NH₄Y activation spectra used for EFA")

# %%
# Follow the eigenvalues as the acquisition window expands.

efa = scp.EFA()
_ = efa.fit(spectra)
forward = efa.f_ev[:, :5]
forward.title = "squared singular value"
_ = forward.T.plot(yscale="log", legend=forward.k.labels, title="Forward EFA")
