# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Smooth a Raman spectrum with two filters
========================================

Use ``Filter`` to apply Savitzky–Golay and Whittaker–Eilers smoothing to
the same Raman spectrum. Compare the traces for noise reduction and band
shape. The :doc:`smoothing tutorial </userguide/processing/smoothing>`
explains how to tune each filter.
"""

import spectrochempy as scp

spectrum = scp.read("ramandata/labspec/SMC1-Initial_RT.txt")
savgol = scp.Filter(method="savgol", size=11, order=2)
whittaker = scp.Filter(method="whittaker", order=2, lamb=10.0)

_ = scp.plot_multiple(
    [spectrum, savgol(spectrum), whittaker(spectrum)],
    method="pen",
    labels=["Measured", "Savitzky–Golay", "Whittaker–Eilers"],
    legend="best",
)
