# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
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


""" Tests for the ndplugin module

"""
import numpy as np
import pandas as pd

from pint import DimensionalityError
from spectrochempy.api import (NDDataset, Axes, Axis,
                                            AxisError, Meta, ur, figure, show)
from spectrochempy.utils import SpectroChemPyWarning
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises, show_do_not_block)


import pytest
import numpy as np
import os


# autosub
#------
@show_do_not_block
def test_autosub(IR_source_1):

    source = IR_source_1

    ranges = [5000., 6000.], [1940., 1820.]


    s1 = source.copy()
    ref = s1[0]

    figure()
    source.plot()
    ref.plot()


    s2 = source.copy()

    s3 = s2.autosub(ref, *ranges)
    # inplace = False
    assert np.round(s2.data[0,0],4) != 0.0000
    assert np.round(s3.data[0,0],4) == 0.0000
    s3.name="varfit"

    figure()
    s3.plot()

    s4 = source.copy()
    s4.autosub(ref, *ranges, method='chi2', inplace=True)
    s4.name = "chi2, inplace"
    assert np.round(s4.data[0,0],4) == 0.0000

    figure()
    s4.plot()  #true avoid blocking due to graphs

    s4 = source.copy()
    from spectrochempy.api import autosub
    s = autosub(s4, ref, *ranges, method='chi2')
    assert np.round(s4.data[0, 0], 4) != 0.0000
    assert np.round(s.data[0, 0], 4) == 0.0000
    s.name = 'chi2 direct call'

    figure()
    s.plot()

    show()

