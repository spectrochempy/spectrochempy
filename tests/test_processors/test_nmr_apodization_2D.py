# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================




""" Tests for the  module

"""
import sys
import functools
import pytest
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises, show_do_not_block)

from spectrochempy.api import *

from spectrochempy.utils import SpectroChemPyWarning


@show_do_not_block
def test_nmr_2D(NMR_source_2D):
    source = NMR_source_2D
    source.plot(nlevels=20)  # , start=0.15)
    show()
    pass


@show_do_not_block
def test_nmr_2D_imag(NMR_source_2D):
    # plt.ion()
    source = NMR_source_2D.copy()
    source.plot(imag=True)
    show()
    pass


@show_do_not_block
def test_nmr_2D_imag_compare(NMR_source_2D):
    # plt.ion()
    source = NMR_source_2D.copy()
    source.plot()
    source.plot(imag=True, cmap='jet', data_only=True, alpha=.3)
    # better not to replot a second colorbar
    show()
    pass


@show_do_not_block
def test_nmr_2D_hold(NMR_source_2D):
    source = NMR_source_2D
    source.plot()
    source.imag.plot(cmap='jet', data_only=True)
    show()
    pass


@show_do_not_block
def test_nmr_2D_em_(NMR_source_2D):
    source = NMR_source_2D.copy()
    source.plot()
    assert source.shape == (96, 948)
    source.em(lb=100. * ur.Hz)
    assert source.shape == (96, 948)
    source.em(lb=50. * ur.Hz, axis=0)
    assert source.shape == (96, 948)
    source.plot(cmap='copper', data_only=True)
    show()
    pass
