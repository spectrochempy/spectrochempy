# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================


""" Tests for the apodization module

"""
import sys
import functools
import pytest
import numpy as np
from spectrochempy.utils.testing import (assert_equal, assert_array_equal, assert_raises,
                         assert_array_almost_equal, assert_equal_units,
                         raises)


from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core import general_preferences as prefs
from spectrochempy.utils import SpectroChemPyWarning, show
from spectrochempy.units import ur

# nmr_processing
# -----------------------------

def test_nmr_1D_show(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    ax1 = dataset.plot()
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

def test_nmr_apodization_with_null(NMR_dataset_1D):
  
    dataset = NMR_dataset_1D.copy()
    
    lb = 0
    arr, apod = dataset.em(lb=lb, inplace=False, retfunc=True)

    # arr and dataset should be equal as no em was applied
    assert_equal(dataset.data, arr.data)
    
    lb = 0.
    gb = 0.
    arr, apod = dataset.gm(lb=lb, gb=gb, inplace=False, retfunc=True)
    
    # arr and dataset should be equal as no em was applied
    assert_equal(dataset.data, arr.data)

def test_nmr_apodization_(NMR_dataset_1D):

    dataset = NMR_dataset_1D.copy()
    
    lb = 100
    arr, apod = dataset.em(lb=lb, inplace=False, retfunc=True)
    
    # arr and dataset should not be equal as inplace=False
    assert not np.array_equal(dataset.data, arr.data)
    assert_array_almost_equal(apod[1],0.9987, decimal=4)

    arr = dataset.em(lb=lb)
    assert_equal(dataset.data, arr.data)
    
    lb = 10.
    gb = 100.
    arr, apod = dataset.gm(lb=lb, gb=gb, inplace=False, retfunc=True)
    
    # arr and dataset should be equal as no em was applied
    assert not np.array_equal(dataset.data, arr.data)
    assert_array_almost_equal(apod[2], 1.000077, decimal=6)

    lb = 10.
    gb = 100.
    arr, apod = dataset.gm(lb=lb, gb=gb, retfunc=True)

    # arr and dataset should be equal as no em was applied
    assert_array_equal(dataset.data, arr.data)
    assert_array_almost_equal(apod[2], 1.000077, decimal=6)

    lb = 10.
    gb = 0.
    arr, apod = dataset.gm(lb=lb, gb=gb, retfunc=True)

    # arr and dataset should be equal as no em was applied
    assert_array_equal(dataset.data, arr.data)
    assert_array_almost_equal(apod[2], 1.000080, decimal=6)


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
#     dataset1.plot(clear=False)
#
#     # dataset1 = dataset1.fft()
#     dataset1.plot()
#     pass


# ----------------------------------------------------------------------------------------------------------------------
def test_nmr_prepare_ipynb():

    # 1D dataset getting function
    import os
    def get_dataset1D():
        dataset1D = NDDataset()
        path = os.path.join(prefs.datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
        dataset1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
        return dataset1D

    # restore original
    dataset1D = get_dataset1D()
    dataset1D = dataset1D[0.:14000.0]

    # normalize amplitude
    dataset1D /= dataset1D.abs().max().values

    # apodize
    LB = 100. * ur.Hz
    apodfunc = dataset1D.em(lb=LB, apply=False)
    lb_dataset = dataset1D.em(lb=LB, inplace=False)  # apply=True by default

    # Plot
    dataset1D.plot(lw=1, color='gray')
    apodfunc.plot(color='r', clear=False)
    lb_dataset.plot(color='r', ls='--', clear=False)

    # shifted
    apodfuncshifted = dataset1D.em(lb=LB, shifted=3000, apply=False)
    apodfuncshifted.plot(color='b', clear=False)
    lbshifted_dataset = dataset1D.em(lb=LB, shifted=3000, inplace=False)  # apply=True by default
    lbshifted_dataset.plot(color='b', ls='--', clear=False)

    # rev
    apodfuncrev = dataset1D.em(lb=LB, rev=True, apply=False)
    apodfuncrev.plot(color='g', clear=False)
    lbrev_dataset = dataset1D.em(lb=LB, rev=True, inplace=False)  # apply=True by default
    lbrev_dataset.plot(xlim=(0, 14000), ylim=(-1, 1), color='g', ls='--', clear=False)


    show()
