# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

from spectrochempy.api import *
from tests.utils import show_do_not_block

import pytest
import os

path = os.path.dirname(os.path.abspath(__file__))

@show_do_not_block
def test_basecor_sequential(IR_source_2D):

    source = IR_source_2D[:5]

    basc = BaselineCorrection(source)
    s = basc([6000.,3500.],[1800.,1500.])

    s.plot()
    show()

@show_do_not_block
def test_basecor_sequential_pchip(IR_source_2D):

    source = IR_source_2D[:5]

    basc = BaselineCorrection(source)
    s = basc([6000., 3500.], [1800., 1500.],
                              interpolation='pchip')
    s.plot()
    show()

@show_do_not_block
def test_basecor_multivariate(IR_source_2D):

    source = IR_source_2D[:5]

    basc = BaselineCorrection(source)
    s = basc([6000., 3500.], [1800., 1500.],
                              method='multivariate',
                              interpolation='polynomial')
    s.plot()
    show()

@show_do_not_block
def test_basecor_multivariate_pchip(IR_source_2D):

    source = IR_source_2D[:5]

    basc = BaselineCorrection(source)
    s = basc([6000., 3500.], [1800., 1500.],
                              method='multivariate',
                              interpolation='pchip')
    s.plot()
    show()

@show_do_not_block
def test_notebook_basecor_bug():
    # coding: utf-8

    source = NDDataset.read_omnic(
        os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))
    source

    # In[3]:


    s = source[:, 1260.0 :5999.0]
    s = s - s[-1]

    # Important note that we use floating point number
    # integer would mean points, not wavenumbers!

    basc = BaselineCorrection(s)

    ranges = [[1261.86, 1285.89], [1556.30, 1568.26], [1795.00, 1956.75],
              [3766.03, 3915.81], [4574.26, 4616.04], [4980.10, 4998.01],
              [5437.52, 5994.70]]  # predifined ranges

    _ = basc.run(*ranges, method='multivariate', interpolation='pchip', npc=5,
                 figsize=(6, 6), zoompreview=4)

    # The regions used to set the baseline are accessible using the `ranges`
    #  attibute:

    # In[ ]:


    ranges = basc.ranges
    print(ranges)

    # In[ ]:


    basc.corrected.plot_stack()


    # In[ ]:




