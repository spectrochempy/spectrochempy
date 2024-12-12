# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""

Fitting 1D dataset
------------------
In this example, we find the least  square solution of a simple linear
equation.

"""
# sphinx_gallery_thumbnail_number = 2

# %%
import os

import spectrochempy as scp

# %%
#  Let's take an IR spectrum

nd = scp.NDDataset.read_omnic(os.path.join("irdata", "nh4y-activation.spg"))

# %%
# and select only the OH region:
ndOH = nd[54, 3800.0:3300.0]
# masking
ndOH[:, 3505.0:3500.0] = scp.MASKED
_ = ndOH.plot()

# %%
# Perform a Fit
# Fit parameters are defined in a script (a single text as below)
script = """
#-----------------------------------------------------------
# syntax for parameters definition:
# name: value, low_bound,  high_bound
# available prefix:
#  # for comments
#  * for fixed parameters
#  $ for variable parameters
#  > for reference to a parameter in the COMMON block
#    (> is forbidden in the COMMON block)
# common block parameters should not have a _ in their names
#-----------------------------------------------------------
#

COMMON:
# common parameters ex.
# $ gwidth: 1.0, 0.0, none
$ gratio: 0.1, 0.0, 1.0

MODEL: LINE_1
shape: asymmetricvoigtmodel
    * ampl:  1.1, 0.0, none
    $ pos:   3620, 3400.0, 3700.0
    $ ratio: 0.0147, 0.0, 1.0
    $ asym: 0.1, 0, 1
    $ width: 50, 0, 1000

MODEL: LINE_2
shape: asymmetricvoigtmodel
    $ ampl:  0.8, 0.0, none
    $ pos:   3540, 3400.0, 3700.0
    > ratio: gratio
    $ asym: 0.1, 0, 1
    $ width: 50, 0, 1000

"""

# %%
# create an Optimize object
f1 = scp.Optimize(log_level="INFO")

# %%
# Show plot and the starting model using the dry parameters (of course it is advisable
# to be as close as possible of a good expectation
f1.script = script

# set dry and continue to show starting model
# reset dry and continue to show starting model
f1.dry = True
f1.autobase = True
f1.fit(ndOH)

# get some information
scp.info_(f"numbers of components: {f1.n_components}")
ndOH.plot()
ax = (f1.components[:]).plot(clear=False)
ax.autoscale(enable=True, axis="y")

# %%
# Now perform a fit with maximum 1000 iterations
f1.max_iter = 1000
_ = f1.fit(ndOH)

# %%
# Show the result
ndOH.plot()
ax = (f1.components[:]).plot(clear=False)
ax.autoscale(enable=True, axis="y")

# %%
# plotmerit
som = f1.inverse_transform()
_ = f1.plotmerit(ndOH, som, method="scatter", markevery=5, markersize=2, lw=2)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
