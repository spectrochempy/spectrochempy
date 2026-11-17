# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Build overlapping synthetic elution profiles
============================================

Build three Gaussian elution profiles with different heights, centers, and
widths. See the :doc:`fitting tutorial </userguide/analysis/fitting>`
for the relationship between the Gaussian parameters and the profile shape.
"""

import spectrochempy as scp

time = scp.Coord.linspace(0.0, 1.0, 200, title="relative elution time")
profiles = scp.NDDataset(
    [
        scp.gaussian(time.data, ampl=1.0, pos=0.25, width=0.235, normalized=False),
        scp.gaussian(time.data, ampl=0.8, pos=0.55, width=0.282, normalized=False),
        scp.gaussian(time.data, ampl=0.6, pos=0.82, width=0.188, normalized=False),
    ],
    coordset=[scp.Coord(labels=["A", "B", "C"], title="component"), time],
    title="relative concentration",
)
_ = profiles.plot_pen(legend=["A", "B", "C"])
