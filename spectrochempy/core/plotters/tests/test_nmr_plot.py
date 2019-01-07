# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================




""" Tests plots for NMR data

"""
import sys
import functools
import pytest
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)


from spectrochempy import *
from spectrochempy.utils import SpectroChemPyWarning

#TODO: complete this tests

# nmr_processing
#-----------------------------

def test_nmr_1D_show(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    ax1 = dataset.plot()
    assert dataset.iscomplex
    show()
    pass

def test_nmr_1D_show_hold(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    # test if we can plot on the same figure
    dataset.plot(xlim=(0.,25000.))
    # we want to superpose a second spectrum
    dataset.plot(imag=True, data_only=True)
    show()


def test_nmr_1D_show_dualdisplay(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    # test if we can plot on the same figure
    dataset.plot(xlim=(0.,25000.))
    dataset.em(lb=100. * ur.Hz)
    # we want to superpose a second spectrum
    dataset.plot()
    show()

    dataset.plot()
    show()


def test_nmr_1D_show_dualdisplay_apodfun(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    # test if we can plot on the same figure
    dataset.plot(xlim=(0.,25000.))
    # we want to superpose a second spectrum wich is the apodization function
    LB = 80 * ur.Hz
    dataset.em(lb=LB)
    dataset.plot(data_only=True)
    # display the apodization function (scaled to the data !)
    apodfun = dataset.em(lb=LB, apply=False)*dataset.data.max()
    apodfun.plot(data_only=True)
    show()


def test_nmr_1D_show_complex(NMR_dataset_1D):
    # display the real and complex at the same time
    dataset = NMR_dataset_1D.copy()
    dataset.plot(show_complex=True, color='green',
                xlim=(0.,30000.), zlim=(-200.,200.))
    show()


def test_nmr_2D(NMR_dataset_2D):
    dataset = NMR_dataset_2D
    dataset.plot(nlevels=20)  # , start=0.15)
    show()
    pass



def test_nmr_2D_imag(NMR_dataset_2D):
    # plt.ion()
    dataset = NMR_dataset_2D.copy()
    dataset.plot(imag=True)
    show()
    pass



def test_nmr_2D_imag_compare(NMR_dataset_2D):
    # plt.ion()
    dataset = NMR_dataset_2D.copy()
    dataset.plot()
    dataset.plot(imag=True, cmap='jet', data_only=True, alpha=.3)
    # better not to replot a second colorbar
    show()
    pass


def test_nmr_2D_hold(NMR_dataset_2D):
    dataset = NMR_dataset_2D
    dataset.plot()
    dataset.imag.plot(cmap='jet', data_only=True)
    show()
    pass

