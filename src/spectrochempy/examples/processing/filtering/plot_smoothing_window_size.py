# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Compare two smoothing windows
=============================

Smooth a noisy synthetic spectrum with two moving averages. Compare
how the narrow and broad bands respond. See the :doc:`smoothing tutorial
</userguide/processing/smoothing>` for guidance on choosing the window.
"""

import spectrochempy as scp

shift = scp.Coord.linspace(100.0, 240.0, 141, title="Raman shift", units="cm^-1")
narrow = scp.gaussian(shift.data, pos=145.0, width=7.0, normalized=False)
broad = scp.gaussian(shift.data, ampl=0.75, pos=198.0, width=30.0, normalized=False)
noise = scp.normal(scale=0.07, size=shift.size, seed=42)
spectrum = scp.NDDataset(
    narrow + broad + noise.data, coordset=[shift], title="intensity"
)

_ = scp.plot_multiple(
    [spectrum, spectrum.smooth(5), spectrum.smooth(21)],
    method="pen",
    labels=["Noisy spectrum", "5 points", "21 points"],
    legend="best",
)
