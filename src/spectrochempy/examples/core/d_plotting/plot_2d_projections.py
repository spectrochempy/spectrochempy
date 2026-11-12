# ruff: noqa: D205, D400, D415
"""
Add projections to a 2D contour plot
====================================

Use ``plot_2D`` to show marginal traces beside a contour plot.
"""

import spectrochempy as scp

x = scp.Coord.linspace(0.0, 10.0, 101, title="direct dimension", units="ppm")
y = scp.Coord.linspace(0.0, 10.0, 81, title="indirect dimension", units="ppm")
direct = scp.gaussian(x.data, pos=6.0, width=1.5, normalized=False)
indirect = scp.gaussian(y.data, pos=4.0, width=2.0, normalized=False)
spectrum = scp.NDDataset(indirect[:, None] * direct, coordset=[y, x])

# %%
# Keep both axes. Each marginal trace sums the intensities along the other
# dimension.
_ = spectrum.plot_2D(method="contour", projections=True)

# %%
# Keep only the x projection and show the maximum along y at each x value.
_ = spectrum.plot_2D(method="contour", projections="x", projection_reducer="max")
