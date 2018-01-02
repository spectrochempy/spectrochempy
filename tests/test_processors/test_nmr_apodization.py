# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================




""" Tests for the  module

"""
import sys
import functools
import pytest
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)


from spectrochempy import *
from spectrochempy.utils import SpectroChemPyWarning


# nmr_processing
#-----------------------------

def test_nmr_1D_show(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    ax1 = dataset.plot()
    assert dataset.is_complex[-1]
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

def test_nmr_em_nothing_calculated(NMR_dataset_1D):
    # em without parameters
    dataset = NMR_dataset_1D.copy()

    arr = dataset.em(apply=False)
    # we should get an array of ones only , as apply = False mean
    # that we do not apply the apodization, but just make the
    # calculation of the apodization function
    assert_equal(arr.data, np.ones_like(dataset.data))

def test_nmr_em_calculated_notapplied(NMR_dataset_1D):
    # em calculated but not applied
    dataset = NMR_dataset_1D.copy()

    lb = 100
    arr = dataset.em(lb=lb, apply=False)
    assert isinstance(arr, NDDataset)

    # here we assume it is 100 Hz
    x = dataset.coordset[-1]
    tc = (1./(lb * ur.Hz)).to(x.units)
    e = np.pi * np.abs(x) / tc
    arrcalc = np.exp(-e.data)

    assert_equal(arr.real.data, arrcalc)  # note that we have to compare
    # to the real part data because of the complex nature of the data

def test_nmr_em_calculated_applied(NMR_dataset_1D):
    # em calculated and applied
    dataset = NMR_dataset_1D.copy()

    lb = 100
    arr = dataset.em(lb=lb, apply=False)

    # here we assume it is 100 Hz
    x = dataset.coordset[-1]
    tc = (1. / (lb * ur.Hz)).to(x.units)
    e = np.pi * np.abs(x) / tc
    arrcalc = np.exp(-e.data)

    # check with apply = True (by default)
    dataset2 = dataset.copy()
    dataset3 = dataset.em(lb=lb)

    # data should be equal
    assert_equal(dataset3.data, (arrcalc*dataset2).data)

    # but also the datasets as whole entity
    dataset4 = arrcalc * dataset2
    dataset3 == dataset4
    assert(dataset3 == dataset4)

def test_nmr_em_calculated_Hz(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()

    lb = 200 * ur.Hz
    x = dataset.coordset[-1]
    tc = (1. / lb).to(x.units)
    e = np.pi * np.abs(x) / tc
    arrcalc = np.exp(-e.data)

    dataset2 = dataset.copy()
    dataset3 = dataset.em(lb=lb, inplace=False)

    # the datasets should be equal
    assert(dataset3 == arrcalc*dataset2)

    # and the original untouched
    assert (dataset != dataset3)

def test_nmr_em_calculated_inplace(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()

    lb = 200 * ur.Hz

    x = dataset.coordset[-1]
    tc = (1. / lb).to(x.units)
    e = np.pi * np.abs(x) / tc
    arrcalc = np.exp(-e.data)

    dataset2 = dataset.copy()
    dataset.em(lb=lb)  # inplace transformation

    # the datasets data array should be equal
    s = arrcalc * dataset2
    assert(np.all(dataset.data == s.data))

    # as well as the whole new datasets
    assert (dataset == arrcalc * dataset2)



def test_nmr_1D_em_(NMR_dataset_1D):

    dataset = NMR_dataset_1D.copy()

    dataset.plot(xlim=(0.,6000.))

    dataset.em(lb=100.*ur.Hz)

    dataset.plot(data_only=True)

    # successive call
    dataset.em(lb=200. * ur.Hz)

    dataset.plot(data_only=True)

    show()


def test_nmr_1D_em_with_no_kw_lb_parameters(NMR_dataset_1D):

    dataset = NMR_dataset_1D.copy()

    dataset.plot()
    dataset.em(100.*ur.Hz, inplace=True)
    dataset.plot()
    show()


def test_nmr_1D_em_inplace(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()

    dataset.plot()
    dataset1 = dataset.em(lb=100. * ur.Hz)
    assert dataset1 is dataset # inplace transform by default
    try:
        assert_array_equal(dataset1.data, dataset.data)
    except AssertionError:
        pass
    show()


def test_nmr_1D_gm(NMR_dataset_1D):

    # first test gm
    dataset = NMR_dataset_1D.copy()

    dataset.plot(xlim=(0.,6000.))

    dataset.gm(lb=100.*ur.Hz, gb=100.*ur.Hz)

    dataset.plot()
    show()

# def test_zf():
#     td = dataset1.meta.td[-1]
#     dataset1 = dataset1.zf(size=2*td)
#     #si = dataset_em.meta.si[-1]
#     dataset1.plot(hold=True)
#
#     # dataset1 = dataset1.fft()
#     dataset1.plot()
#     pass

