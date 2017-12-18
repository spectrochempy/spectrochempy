# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================




""" Tests for the ndplugin module

"""
import numpy as np
import pandas as pd

from spectrochempy.api import *
from spectrochempy.utils import SpectroChemPyWarning
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)

import pytest
import numpy as np
import os


# autosub
#------

def test_autosub(IR_source_2D):

    source = IR_source_2D

    ranges = [5000., 6000.], [1940., 1820.]


    s1 = source.copy()
    ref = s1[0]

    source.plot_stack()

    ref.plot()
    #show()

    s2 = source.copy()

    s3 = s2.autosub(ref, *ranges)
    # inplace = False
    assert np.round(s2.data[0,0],4) != 0.0000
    assert np.round(s3.data[0,0],4) == 0.0000
    s3.name="varfit"

    s3.plot_stack()

    s4 = source.copy()
    s4.autosub(ref, *ranges, method='chi2', inplace=True)
    s4.name = "chi2, inplace"
    assert np.round(s4.data[0,0],4) == 0.0000

    s4.plot_stack()  #true avoid blocking due to graphs

    s4 = source.copy()
    s = autosub(s4, ref, *ranges, method='chi2')
    assert np.round(s4.data[0, 0], 4) != 0.0000
    assert np.round(s.data[0, 0], 4) == 0.0000
    s.name = 'chi2 direct call'

    s.plot_stack()

    show()

