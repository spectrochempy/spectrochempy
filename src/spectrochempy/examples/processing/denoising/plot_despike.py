# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Remove spikes from a Raman spectrum
===================================

Remove sharp positive spikes from one measured Raman spectrum and compare
the result with the input. See the :doc:`denoising tutorial
</userguide/processing/denoising>` for choosing a detection threshold and
checking that narrow Raman bands survive the correction.
"""

import spectrochempy as scp

series = scp.read("ramandata/labspec/serie190214-1.txt")
spectrum = series[0, 70.0:].snip()
cleaned = spectrum.despike(size=11, delta=5)

_ = scp.plot_multiple(
    [spectrum, cleaned],
    method="pen",
    labels=["Before despiking", "After despiking"],
    legend="best",
)
