# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Correct offsets and gains with MSC
==================================

Apply multiplicative scatter correction to three synthetic spectra that
share two overlapping bands but differ in offset and gain. The corrected
curves overlap because these are the only differences in the input.
See the :doc:`transformations tutorial </userguide/processing/transformations>`
for choosing a reference.
"""

import spectrochempy as scp

x = scp.Coord.linspace(1100.0, 1300.0, 201, title="wavenumber", units="cm^-1")
first = scp.gaussian(x.data, pos=1180.0, width=30.0, normalized=False)
second = scp.gaussian(x.data, pos=1210.0, width=40.0, ampl=0.6, normalized=False)
reference = scp.NDDataset(first + second, coordset=[x], title="intensity")
spectra = scp.stack(0.7 * reference + 0.1, reference, 1.3 * reference + 0.2)

corrected = spectra.msc(reference=reference)
_ = scp.multiplot(
    [spectra, corrected],
    method="lines",
    ncol=2,
    labels=["Different offsets and gains", "After MSC"],
    sharey=True,
    figsize=(10, 4),
)
