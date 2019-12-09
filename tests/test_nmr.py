# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================


""" Tests for the nmr related processors

"""

import pytest

import os
import numpy as np
from spectrochempy.utils.testing import (assert_equal, assert_array_equal, assert_raises,
                         assert_array_almost_equal, assert_equal_units, raises)


from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import general_preferences as prefs
from spectrochempy.utils import show
from spectrochempy.units import ur

# 1D

## Reader

def test_nmr_reader_1D():
    path = os.path.join(prefs.datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
    
    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    assert ndd.__str__() == 'NDDataset: [complex128] unitless (size: 12411)'
    assert "<tr><td style='padding-right:5px; padding-bottom:0px; padding-top:0px; width:124px'><font color='green'> " \
           " coordinates</font> </td><td style='text-align:left; padding-bottom:0px; padding-top:0px; border:.5px " \
           "solid lightgray;  '> <div><font color='blue'>[       0        4 ... 4.964e+04 4.964e+04] us</font></div>" \
           "</td><tr>" in ndd._repr_html_()


## plot

def test_nmr_1D_show(NMR_dataset_1D):
    # test simple plot
    
    dataset = NMR_dataset_1D.copy()
    ax1 = dataset.plot()
    

def test_nmr_1D_show_complex(NMR_dataset_1D):
    
    dataset = NMR_dataset_1D.copy()
    dataset.plot(xlim=(0.,25000.))
    dataset.plot(imag=True, color = 'r', data_only=True, clear=False)
    
    # display the real and complex at the same time
    dataset.plot(show_complex=True, color='green',
                xlim=(0.,30000.), zlim=(-200.,200.))
    show()

## apodization

def test_nmr_1D_apodization(NMR_dataset_1D):
  
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

    dataset = NMR_dataset_1D.copy()
    dataset.plot(xlim=(0.,6000.))

    dataset.em(lb=100.*ur.Hz)
    dataset.plot(c='r', data_only=True, clear=False)

    # successive call
    dataset.em(lb=200. * ur.Hz)
    dataset.plot(c='g', data_only=True, clear=False)

    dataset = NMR_dataset_1D.copy()
    dataset.plot()
    
    dataset.em(100.*ur.Hz)
    dataset.plot(c='r', data_only=True, clear=False)

    # first test gm
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()  # normalize
    dataset.plot(xlim=(0.,6000.))

    dataset, apod = dataset.gm(lb=-100.*ur.Hz, gb=100.*ur.Hz, retfunc=True)
    dataset.plot(c='b', data_only=True, clear=False)
    apod.plot(c='r', data_only=True, clear=False)
    show()


## FFT

def test_nmr_fft_1D(NMR_dataset_1D):
    
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize
    
    dataset1D.x.ito('s')
    new = dataset1D.fft(tdeff=8192, size=2**15)
    
    new2 = new.ifft(inplace=False)

    dataset1D.plot()
    (new2-.1).plot(color='r', clear=False)
    
    new.plot()
    
    show()

##### 2D NMR ########

def test_nmr_reader_2D():
    path = os.path.join(prefs.datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_2d')
    
    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    assert ndd.__str__() == "NDDataset: [quaternion] unitless (shape: (y:96, x:948))"
    assert "<tr><td style='padding-right:5px; padding-bottom:0px; padding-top:0px; width:124px'><font color='green'>" \
           "       values</font> </td><td style='text-align:left; padding-bottom:0px; padding-top:0px; border:.5px " \
           "solid lightgray;  '> <div><font color='blue'>         RR[[ 0.06219   0.1467 ...  0.04565  0.03068]<br/>  " \
           "          [-0.05969 -0.08752 ... -0.05134 -0.05994]<br/>            ...<br/>            [       0        " \
           "0 ...        0        0]<br/>            [       0        0 ...        0        0]]<br/>         " \
           "RI[[  0.2238   0.1985 ...   0.1662 -0.03262]<br/>            [0.006566  -0.0282 ...  0.02949  0.06717]" \
           "<br/>            ...<br/>            [      -0       -0 ...       -0       -0]<br/>            [      -0" \
           "       -0 ...       -0       -0]]<br/>         IR[[-0.003312 -0.001535 ...  0.02067 -0.08058]<br/>" \
           "            [-0.05685   0.1174 ...  0.05831 -0.003414]<br/>            ...<br/>            [       0 " \
           "       0 ...        0        0]<br/>            [       0        0 ...        0        0]]<br/> " \
           "        II[[  0.1623   0.0563 ... -0.02654  0.01094]<br/>            [ -0.1344 0.006515 ...  " \
           "0.08239 -0.00516]<br/>            ...<br/>            [      -0       -0 ...       -0       -0]<br/> " \
           "           [      -0       -0 ...       -0       -0]]</font></div></td><tr>" in ndd._repr_html_()


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
    dataset.plot(imag=True, cmap='jet', data_only=True, alpha=.3, clear=False)
    # better not to replot a second colorbar
    show()
    pass


def test_nmr_2D_hold(NMR_dataset_2D):
    dataset = NMR_dataset_2D
    dataset.plot()
    dataset.imag.plot(cmap='jet', data_only=True, clear=False)
    show()
    pass

# apodization

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


def test_nmr_2D(NMR_dataset_2D):
    dataset = NMR_dataset_2D
    dataset.plot(nlevels=20)  # , start=0.15)
    show()
    pass




