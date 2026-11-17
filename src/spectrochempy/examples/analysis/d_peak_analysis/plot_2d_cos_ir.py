# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Plot synchronous and asynchronous 2D correlation maps
=====================================================

Calculate 2D-COS for two synthetic infrared bands that vary over time.
Red solid contours are positive and blue dashed contours are negative.
See the :doc:`2D-COS tutorial </userguide/analysis/correlation_spectroscopy>`
for experimental preparation and interpretation.
"""

import spectrochempy as scp

time = scp.Coord.linspace(0.0, 10.0, 21, title="time", units="s")
x = scp.Coord.linspace(1800.0, 1400.0, 101, title="wavenumber", units="cm^-1")
band_a = scp.gaussian(x.data, pos=1680.0, width=30.0, normalized=False)
band_b = scp.gaussian(x.data, pos=1540.0, width=35.0, normalized=False)
early = time.data / (1.0 + time.data)
late = time.data / (5.0 + time.data)
spectra = scp.NDDataset(
    early[:, None] * band_a + late[:, None] * band_b,
    coordset=[time, x],
    title="absorbance",
    units="absorbance",
)

result = scp.correlation_spectroscopy(spectra, perturbation_dim="y")
_ = scp.multiplot(
    [result.synchronous, result.asynchronous],
    method="contour",
    ncol=2,
    labels=["Synchronous", "Asynchronous"],
    positive_color="#b2182b",
    negative_color="#2166ac",
    negative_linestyle="--",
    colorbar=False,
    figsize=(9, 4.5),
)
