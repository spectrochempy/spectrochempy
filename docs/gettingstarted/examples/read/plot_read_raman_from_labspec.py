# -*- coding: utf-8 -*-
# %%
# flake8: noqa
# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
Loading RAMAN experimental file
============================================

Here we load experimental LABSPEC spectra and plot them.

"""

import spectrochempy as scp

# %%
# define the folder where are the spectra
datadir = scp.preferences.datadir
ramandir = datadir / "ramandata"

# %%

A = scp.read_labspec("Activation.txt", directory=ramandir)
A.plot()

A = scp.read_labspec("532nm-191216-Si_200mu.txt", directory=ramandir)
A.plot()

A = scp.read_labspec("serie190214-1.txt", directory=ramandir)
A.plot(colorbar=True)
A.plot_map(colorbar=True)

A = scp.read_labspec("SMC1-Initial_RT.txt", directory=ramandir)
A.plot()

# %%
# Open a dialog - note the presence of the keyword directory
B = scp.read_labspec(directory=ramandir)

# %%
# this pack all spectra of the subdir directory (without dialog - look at the difference above)
B = scp.read_labspec(ramandir / "subdir")
B.plot()

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
