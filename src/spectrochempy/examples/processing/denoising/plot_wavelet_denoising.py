# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Denoise a spectrum with wavelets
================================

Use the optional ``wavelets`` extra to reduce noise in two overlapping
synthetic Raman bands. The clean signal provides a reference for judging
how well the filter preserves band shape. See the :doc:`wavelet guide
</userguide/processing/wavelet_denoising>` for the threshold and input requirements.
"""

import spectrochempy as scp

shift = scp.Coord.linspace(1000.0, 1128.0, 513, title="Raman shift", units="cm^-1")
band_a = scp.gaussian(shift.data, pos=1054.0, width=8.0, normalized=False)
band_b = scp.gaussian(shift.data, pos=1064.0, width=12.0, ampl=0.65, normalized=False)
noise = scp.normal(scale=0.09, size=shift.size, seed=42)
reference = scp.NDDataset(band_a + band_b, coordset=[shift], title="intensity")
noisy = reference + noise.data

filtered = scp.Filter(method="wavelet").transform(noisy, dim="x")
_ = scp.plot_multiple(
    [noisy, filtered, reference],
    method="pen",
    labels=["Noisy signal", "Wavelet filtered", "Clean reference"],
    ls=["-", "-", "--"],
    legend="best",
)
