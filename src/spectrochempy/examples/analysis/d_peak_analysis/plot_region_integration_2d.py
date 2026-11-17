# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Integrate a peak in a 2D spectrum
=================================

Integrate a synthetic peak inside a box defined by pump and probe wavenumbers.
See the :doc:`peak integration tutorial </userguide/analysis/peak_integration>`
for bounds between samples and uncertainty from a separate background region.
"""

import spectrochempy as scp

x = scp.Coord.linspace(1950, 2150, 101, units="cm^-1", title="pump wavenumber")
y = scp.Coord.linspace(1950, 2150, 81, units="cm^-1", title="probe wavenumber")
probe = scp.gaussian(y.data, pos=2040, width=20, normalized=False)
pump = scp.gaussian(x.data, pos=2070, width=25, normalized=False)
signal = scp.NDDataset(probe[:, None] * pump, coordset=[y, x], units="absorbance")

bounds = {"x": (2037, 2103), "y": (2012, 2069)}
result = signal.integrate_region(bounds)

ax = signal.plot_image(figsize=(7, 5), cmap="viridis", colorbar=True)
left, right = bounds["x"]
bottom, top = bounds["y"]
_ = ax.plot(
    [left, right, right, left, left],
    [bottom, bottom, top, top, bottom],
    color="black",
    linewidth=1.5,
)
_ = ax.set_title(f"Integrated volume: {result.value:.1f} {result.unit:~P}")
