# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
IRIS: 2D-IRIS analysis (plugin)
=================================

This example introduces the 2D-IRIS analysis provided by the optional
``spectrochempy-iris`` plugin.
"""

# %%
from importlib.util import find_spec

import spectrochempy as scp

if find_spec("spectrochempy_iris") is None:
    raise ImportError(
        "This example requires the optional spectrochempy-iris plugin.\n"
        "Install it with: pip install spectrochempy[iris]"
    )

# %%
# The IRIS plugin provides the :class:`~spectrochempy_iris.IRIS` class
# and related tools. A typical workflow is shown in the
# :ref:`quickstart <quickstart>`.

from spectrochempy_iris import IRIS
from spectrochempy_iris import IrisKernel

# Load CO adsorption data
ds = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2250.0:1950.0]

# Attach pressure coordinates
pressure = [
    0.003, 0.004, 0.009, 0.014, 0.021, 0.026, 0.036, 0.051,
    0.093, 0.150, 0.203, 0.300, 0.404, 0.503, 0.602, 0.702,
    0.801, 0.905, 1.004,
]
ds.y = scp.Coord(pressure, title="Pressure", units="torr")

print(f"Loaded dataset: {ds}")

# %%
# Run IRIS analysis:

iris = IRIS(reg_par=[-10, 1, 12])
K = IrisKernel(ds, "langmuir", q=[-7, -1, 50])
iris.fit(ds, K)

_ = iris.f[-7].plot_contour(colorbar=True)

# scp.show()
