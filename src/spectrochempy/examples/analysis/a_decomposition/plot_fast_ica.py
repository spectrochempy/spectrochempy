# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Separate two mixed signals with FastICA
=======================================

Separate a smooth oscillation and a switching signal from two mixtures.
See the :doc:`FastICA tutorial </userguide/analysis/fast_ica>` for the
orientation of the input matrix and how to interpret the sources.
"""

import numpy as np

import spectrochempy as scp

time = scp.Coord.linspace(0.0, 6.0, 601, title="time", units="s")
smooth = scp.sin(2.0 * np.pi * time.data)
switching = (scp.sin(5.0 * np.pi * time.data) >= 0) * 2.0 - 1.0
mixtures = scp.NDDataset(
    [smooth + 0.6 * switching, 0.4 * smooth + switching],
    coordset=[scp.Coord(labels=["Mixture 1", "Mixture 2"]), time],
    title="relative signal",
)
model = scp.FastICA(n_components=2, random_state=42)
sources = model.fit_transform(mixtures.T).T
sources.title = "relative signal"

_ = scp.multiplot(
    [mixtures, sources],
    labels=["Observed mixtures", "Estimated sources"],
    ncol=2,
    method="pen",
    figsize=(10, 4),
)
