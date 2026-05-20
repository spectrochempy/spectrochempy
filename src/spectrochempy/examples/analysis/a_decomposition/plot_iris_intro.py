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
from spectrochempy.examples._plugin_deps import requires_plugin

import spectrochempy as scp

if requires_plugin("spectrochempy_iris", "spectrochempy-iris"):
    # %%
    # The IRIS plugin provides package-level workflows under ``scp.iris`` and
    # dataset-bound helpers under ``dataset.iris``.

    # Load CO adsorption data
    ds = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2250.0:1950.0]

    # Attach pressure coordinates
    pressure = [
        0.003, 0.004, 0.009, 0.014, 0.021, 0.026, 0.036, 0.051, 0.093, 0.150,
        0.203, 0.300, 0.404, 0.503, 0.602, 0.702, 0.801, 0.905, 1.004,
    ]
    ds.y = scp.Coord(pressure, title="Pressure", units="torr")

    print(f"Loaded dataset: {ds}")

    # %%
    # Build the IRIS kernel from the dataset accessor, then run the analysis:

    K = ds.iris.kernel_matrix(kernel_type="langmuir", q=[-7, -1, 50])
    iris_analysis = scp.iris.IRIS(reg_par=[-10, 1, 12])
    iris_analysis.fit(ds, K)

    _ = iris_analysis.f[-7].plot_contour(colorbar=True)

# scp.show()
