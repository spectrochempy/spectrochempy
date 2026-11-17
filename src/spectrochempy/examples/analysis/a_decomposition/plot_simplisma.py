# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Estimate component spectra with SIMPLISMA
=========================================

Estimate up to four component spectra from an HPLC-DAD run using pure
variable selection on a non-negative copy of the data. See the :doc:`MCR-ALS tutorial </userguide/analysis/mcr_als>`
for the selection settings and how to use the spectra as initial estimates.
"""

import spectrochempy as scp

datasets = scp.read_matlab("matlabdata/als2004dataset.MAT", merge=False)
chromatogram = next(dataset for dataset in datasets if dataset.name == "m1")

model = scp.SIMPLISMA(n_components=4, noise=3, tol=0.05)
_ = model.fit(chromatogram.clip(0))
spectra = model.result.components
spectra.set_coordset(
    None, scp.Coord.arange(spectra.shape[-1], title="spectral channel")
)
spectra.title = "relative intensity"
labels = [f"Component {i + 1}" for i in range(spectra.shape[0])]
_ = spectra.plot_pen(legend=labels, legend_loc="upper right")
