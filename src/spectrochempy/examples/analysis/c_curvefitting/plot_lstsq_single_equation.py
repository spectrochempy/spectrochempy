# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Fit an absorbance calibration line
==================================

Fit a line to synthetic absorbance measurements using the concentration
coordinate as the predictor. See the :doc:`fitting tutorial </userguide/analysis/fitting>`
for alternative inputs, coefficient units, and residuals.
"""

import spectrochempy as scp

concentration = scp.Coord([0, 2, 4, 6, 8], title="concentration", units="mmol/L")
absorbance = scp.NDDataset(
    [0.015, 0.159, 0.329, 0.466, 0.624], coordset=[concentration], title="absorbance"
)

model = scp.LSTSQ()
_ = model.fit(absorbance)
fitted = model.predict()

ax = absorbance.plot_scatter(label="Calibration points")
_ = fitted.plot_pen(ax=ax, clear=False, color="red", label="Fitted line", legend=True)
