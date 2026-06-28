# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Built-in line-shape helpers
---------------------------

This example shows the direct top-level helpers for generating common 1D
profiles without instantiating fitting models explicitly.
"""

# %%
import spectrochempy as scp

# %%
# Build a shared abscissa and a few synthetic profiles.
x = scp.linspace(-4.0, 4.0, 400)

profiles = [
    scp.gaussian(x, ampl=1.0, pos=-1.5, width=0.9),
    scp.lorentzian(x, ampl=1.0, pos=0.0, width=0.8),
    scp.voigt(x, ampl=1.0, pos=1.5, width=1.0, ratio=0.5),
    scp.asymmetricvoigt(x, ampl=1.0, pos=2.5, width=1.2, ratio=0.4, asym=0.25),
]

# %%
# Stack them as columns to compare the available shapes side by side.
shapes = scp.stack(profiles, axis=1)
shapes.x.title = "time"
shapes.y = scp.Coord(labels=["gaussian", "lorentzian", "voigt", "asym. voigt"])

# %%
ax = shapes.T.plot()
_ = ax.legend(shapes.y.labels)

# %%
