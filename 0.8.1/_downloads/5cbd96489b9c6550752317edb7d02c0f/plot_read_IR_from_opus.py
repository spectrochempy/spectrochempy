# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Loading Bruker OPUS files
=========================

Here we load an experimental Bruker OPUS files and plot it.

"""
# %%

import spectrochempy as scp

Z = scp.read_opus(
    ["test.0000", "test.0001", "test.0002", "test.0003"], directory="irdata/OPUS"
)
Z

# %%
# plot it

Z.plot()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python
#
# scp.show()
