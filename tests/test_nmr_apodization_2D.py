# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================




""" Tests for the  module

"""
import sys
import functools
import pytest
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)

from spectrochempy import *

from spectrochempy.utils import SpectroChemPyWarning



#@pytest.mark.skip('not yet finish')
def test_nmr_2D_em_x(NMR_dataset_2D):
    
    dataset = NMR_dataset_2D.copy()
    assert dataset.shape == (96, 948)
    dataset.plot_map()  # plot original

    dataset = NMR_dataset_2D.copy()
    dataset.plot_map()
    dataset.em(lb=50. * ur.Hz, axis=-1)
    assert dataset.shape == (96, 948)
    dataset.plot_map(cmap='copper', data_only=True, clear=False)  # em on dim=x

    dataset = NMR_dataset_2D.copy()
    dataset.plot_map()
    dataset.em(lb=50. * ur.Hz, dim='x')
    assert dataset.shape == (96, 948)
    dataset.plot_map(cmap='copper', data_only=True, clear=False)  # em on dim=x

    show()
    pass

def test_nmr_2D_em_y(NMR_dataset_2D):
    
    dataset = NMR_dataset_2D.copy()
    assert dataset.shape == (96, 948)
    dataset.plot_map()  # plot original
    
    dataset = NMR_dataset_2D.copy()
    dataset.plot_map()
    dataset.em(lb=50. * ur.Hz, dim=0)
    assert dataset.shape == (96, 948)
    dataset.plot_map(cmap='copper', data_only=True, clear=False)  # em on dim=x
    
    dataset = NMR_dataset_2D.copy()
    dataset.plot_map()
    dataset.em(lb=50. * ur.Hz, dim='y')
    assert dataset.shape == (96, 948)
    dataset.plot_map(cmap='copper', data_only=True, clear=False)  # em on dim=x
    
    show()
    pass