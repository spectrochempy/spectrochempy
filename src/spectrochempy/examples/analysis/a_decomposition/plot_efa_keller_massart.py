# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Detect two overlapping chromatographic factors
==============================================

Fit EFA to a simulated chromatogram based on :cite:t:`keller:1991`.
Compare the first three eigenvalue curves in both directions.
See the :doc:`EFA tutorial </userguide/analysis/efa>` for interpretation.
"""

import spectrochempy as scp

time = scp.Coord.arange(15, units="min", title="elution time")
profiles = scp.zeros((2, time.size), coordset=[None, time])
profiles[0, 3:8] = [1, 3, 6, 3, 1]
profiles[1, 5:11] = [1, 3, 5, 3, 1, 0.5]
spectra = scp.NDDataset([[2.0, 3.0, 4.0, 2.0], [3.0, 4.0, 2.0, 1.0]])
chromatogram = scp.dot(profiles.T, spectra)
chromatogram += scp.normal(scale=0.1, size=chromatogram.shape, seed=42)

efa = scp.EFA()
_ = efa.fit(chromatogram)
forward = efa.f_ev[:, :3]
backward = efa.b_ev[:, :3]
forward.title = backward.title = "squared singular value"
_ = scp.multiplot(
    [scp.log10(forward.clip(1e-4)).T, scp.log10(backward.clip(1e-4)).T],
    labels=["Forward EFA", "Backward EFA"],
    ncol=2,
    method="pen",
    legend=["First", "Second", "Third"],
    sharey=True,
    figsize=(10, 4),
)
