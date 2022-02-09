#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

"""
Load and save NDDataset
=======================

To import data from different software, there is several `readers` taht can be used.
In this example, we show how to load and save data in a SpectroChempy format.

"""

import spectrochempy as scp

datadir = scp.preferences.datadir
dataset = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")

# %%
# Display content:

dataset._repr_html_()

scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
