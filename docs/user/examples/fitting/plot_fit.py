# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""

Fitting 1D dataset
------------------
In this example, we find the least  square solution of a simple linear
equation.

"""
# sphinx_gallery_thumbnail_number = 2

import spectrochempy as scp
import os

########################################################################
#  Let's take an IR spectrum

nd = scp.NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))

########################################################################
# where we select only region (OH region)

ndOH = nd[54, 3700.:3400.]

ndOH.plot()

########################################################################
## Perform a Fit
#  Fit parameters are defined in a script (a single text as below)


script= """
#-----------------------------------------------------------
# syntax for parameters definition:
# name: value, low_bound,  high_bound
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
shape: assymvoigtmodel
    * ampl:  1.0, 0.0, none
    $ pos:   3620, 3400.0, 3700.0
    $ ratio: 0.0147, 0.0, 1.0
    $ assym: 0.1, 0, 1
    $ width: 200, 0, 1000

MODEL: LINE_2
shape: assymvoigtmodel
    $ ampl:  0.2, 0.0, none
    $ pos:   3520, 3400.0, 3700.0
    > ratio: gratio
    $ assym: 0.1, 0, 1
    $ width: 200, 0, 1000
        
"""


##############################################################################
# create a fit object

f1 = scp.Fit(ndOH, script, silent=True)

##############################################################################
# Show plot and the starting model before the fit (of course it is advisable
# to be as close as possible of a good expectation

f1.dry_run()

ndOH.plot(plot_model=True)

f1.run(maxiter=1000)

##############################################################################
# Show the result after 1000 iterations

ndOH.plot(plot_model=True)

#show() # uncomment to show plot if needed()