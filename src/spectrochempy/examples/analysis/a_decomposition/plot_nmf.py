# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Recover two non-negative spectral components
============================================

Fit NMF to synthetic spectra whose two overlapping bands contribute in
different proportions across the series.
See the :doc:`NMF tutorial </userguide/analysis/nmf>` for data preparation,
rank comparison, and reconstruction.
"""

import spectrochempy as scp

time = scp.Coord.linspace(0.0, 10.0, 21, title="time", units="s")
x = scp.Coord.linspace(1740.0, 1510.0, 121, title="wavenumber", units="cm^-1")
band_a = scp.gaussian(x.data, pos=1640.0, width=50.0, normalized=False)
band_b = scp.gaussian(x.data, pos=1605.0, width=50.0, normalized=False)
fraction = 0.05 + 0.09 * time.data[:, None]
spectra = scp.NDDataset(
    (1.0 - fraction) * band_a + fraction * band_b,
    coordset=[time, x],
    title="absorbance",
)

model = scp.NMF(n_components=2, init="nndsvda", random_state=0, max_iter=1000)
_ = model.fit(spectra)
components = model.result.components
_ = scp.multiplot(
    [spectra[::5], components],
    method="pen",
    ncol=2,
    labels=["Mixture spectra", "Estimated components"],
    figsize=(10, 3.5),
)
