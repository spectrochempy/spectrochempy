# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os

import numpy as np
import pytest

from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur
from spectrochempy.utils.plots import show
from spectrochempy.utils.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

pytestmark = pytest.mark.skip("all tests still WIP")


# 1D

# Reader


def test_nmr_reader_1D():
    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_1d"
    )

    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_topspin(path, expno=1, remove_digital_filter=True)
    assert ndd.__str__() == "NDDataset: [complex128] unitless (size: 12411)"
    assert (
        "<tr><td style='padding-right:5px; padding-bottom:0px; padding-top:0px; width:124px'><font color='#28A745'> "
        " coordinates</font> </td><td style='text-align:left; padding-bottom:0px; padding-top:0px; border:.5px "
        "solid lightgray;  '> <div><font color='#2D7FF9'>[       0        4 ... 4.964e+04 4.964e+04] us</font></div>"
        "</td><tr>" in ndd._repr_html_()
    )


# plot


def test_nmr_1D_show(NMR_dataset_1D):
    # test simple plot

    dataset = NMR_dataset_1D.copy()
    dataset.plot()

    show()


def test_nmr_1D_show_complex(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset.plot(xlim=(0.0, 25000.0))
    dataset.plot(imag=True, color="r", data_only=True, clear=False)

    # display the real and complex at the same time
    dataset.plot(
        show_complex=True, color="green", xlim=(0.0, 30000.0), zlim=(-200.0, 200.0)
    )
    show()


# apodization
def test_nmr_1D_apodization(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()  # normalize

    lb = 0
    arr, apod = dataset.em(lb=lb, retapod=True)

    # arr and dataset should be equal as no em was applied
    assert_equal(dataset.data, arr.data)

    lb = 0.0
    gb = 0.0
    arr, apod = dataset.gm(lb=lb, gb=gb, retapod=True)

    # arr and dataset should be equal as no em was applied
    assert_equal(dataset.data, arr.data)

    lb = 100
    arr, apod = dataset.em(lb=lb, retapod=True)

    # arr and dataset should not be equal as inplace=False
    assert not np.array_equal(dataset.data, arr.data)
    assert_array_almost_equal(apod[1], 0.9987, decimal=4)

    # inplace=True
    dataset.plot(xlim=(0.0, 6000.0))

    dataset.em(lb=100.0 * ur.Hz, inplace=True)
    dataset.plot(c="r", data_only=True, clear=False)

    # successive call
    dataset.em(lb=200.0 * ur.Hz, inplace=True)
    dataset.plot(c="g", data_only=True, clear=False)

    dataset = NMR_dataset_1D.copy()
    dataset.plot()

    dataset.em(100.0 * ur.Hz, inplace=True)
    dataset.plot(c="r", data_only=True, clear=False)

    # first test gm
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()  # normalize

    dataset.plot(xlim=(0.0, 6000.0))

    dataset, apod = dataset.gm(lb=-100.0 * ur.Hz, gb=100.0 * ur.Hz, retapod=True)
    dataset.plot(c="b", data_only=True, clear=False)
    apod.plot(c="r", data_only=True, clear=False)

    show()


# FFT


def test_nmr_fft_1D(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize
    dataset1D.x.ito("s")
    new = dataset1D.fft(tdeff=8192, size=2**15)
    new.plot()
    new2 = new.ifft()
    dataset1D.plot()
    (new2 - 1.0).plot(color="r", clear=False)
    show()


def test_nmr_fft_1D_our_Hz(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize
    LB = 10.0 * ur.Hz
    GB = 50.0 * ur.Hz
    dataset1D.gm(gb=GB, lb=LB)
    new = dataset1D.fft(size=32000, ppm=False)
    new.plot(xlim=[5000, -5000])
    show()


def test_nmr_manual_1D_phasing(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize

    dataset1D.em(10.0 * ur.Hz)  # inplace broadening
    transf = dataset1D.fft(tdeff=8192, size=2**15)  # fft
    transf.plot()  # plot)

    # manual phasing
    transfph = transf.pk(verbose=True)  # by default pivot = 'auto'
    transfph.plot(xlim=(20, -20), clear=False, color="r")
    assert_array_equal(transfph.data, transf.data)  # because phc0 already applied

    transfph3 = transf.pk(pivot=50, verbose=True)
    transfph3.plot(clear=False, color="r")
    not assert_array_equal(
        transfph3.data, transfph.data
    )  # because phc0 already applied
    #
    transfph4 = transf.pk(pivot=100, phc0=40.0, verbose=True)
    transfph4.plot(xlim=(20, -20), clear=False, color="g")
    assert transfph4 != transfph

    transfph4 = transf.pk(pivot=100, verbose=True, inplace=True)
    (transfph4 - 10).plot(xlim=(20, -20), clear=False, color="r")

    show()


def test_nmr_auto_1D_phasing():
    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_1d"
    )
    ndd = NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(10.0 * ur.Hz, inplace=True)
    transf = ndd.fft(tdeff=8192, size=2**15)
    transf.plot(xlim=(20, -20), ls=":", color="k")

    transfph2 = transf.pk(verbose=True)
    transfph2.plot(xlim=(20, -20), clear=False, color="r")

    # automatic phasing
    transfph3 = transf.apk(verbose=True)
    (transfph3 - 1).plot(xlim=(20, -20), clear=False, color="b")

    transfph4 = transf.apk(algorithm="acme", verbose=True)
    (transfph4 - 2).plot(xlim=(20, -20), clear=False, color="g")

    transfph5 = transf.apk(algorithm="neg_peak", verbose=True)
    (transfph5 - 3).plot(xlim=(20, -20), clear=False, ls="-", color="r")

    transfph6 = transf.apk(algorithm="neg_area", verbose=True)
    (transfph6 - 4).plot(xlim=(20, -20), clear=False, ls="-.", color="m")

    transfph4 = transfph6.apk(algorithm="acme", verbose=True)
    (transfph4 - 6).plot(xlim=(20, -20), clear=False, color="b")

    show()


def test_nmr_multiple_manual_1D_phasing():
    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_1d"
    )
    ndd = NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(10.0 * ur.Hz)  # inplace broadening

    transf = ndd.fft(tdeff=8192, size=2**15)

    transfph1 = transf.pk(verbose=True)
    transfph1.plot(xlim=(20, -20), color="k")

    transfph2 = transf.pk(verbose=True)
    transfph2.plot(xlim=(20, -20), clear=False, color="r")

    transfph3 = transf.pk(52.43836, -16.8366, verbose=True)
    transfph3.plot(xlim=(20, -20), clear=False, color="b")

    show()


def test_nmr_multiple_auto_1D_phasing():
    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_1d"
    )
    ndd = NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(10.0 * ur.Hz)  # inplace broadening

    transf = ndd.fft(tdeff=8192, size=2**15)
    transf.plot(xlim=(20, -20), ls=":", color="k")

    t1 = transf.apk(algorithm="neg_peak", verbose=True)
    (t1 - 5.0).plot(xlim=(20, -20), clear=False, color="b")

    t2 = t1.apk(algorithm="neg_area", verbose=True)
    (t2 - 10).plot(xlim=(20, -20), clear=False, ls="-.", color="m")

    t3 = t2.apk(algorithm="acme", verbose=True)
    (t3 - 15).plot(xlim=(20, -20), clear=False, color="r")

    show()


# #### 2D NMR ########


def test_nmr_reader_2D():
    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_2d"
    )

    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_topspin(path, expno=1, remove_digital_filter=True)
    assert ndd.__str__() == "NDDataset: [quaternion] unitless (shape: (y:96, x:948))"
    assert (
        "<tr><td style='padding-right:5px; padding-bottom:0px; padding-top:0px;"
        in ndd._repr_html_()
    )


def test_nmr_2D_imag(NMR_dataset_2D):
    # plt.ion()
    dataset = NMR_dataset_2D.copy()
    dataset.plot(imag=True)
    show()


def test_nmr_2D_imag_compare(NMR_dataset_2D):
    # plt.ion()
    dataset = NMR_dataset_2D.copy()
    dataset.plot()
    dataset.plot(imag=True, cmap="jet", data_only=True, alpha=0.3, clear=False)
    # better not to replot a second colorbar
    show()


def test_nmr_2D_hold(NMR_dataset_2D):
    dataset = NMR_dataset_2D
    dataset.plot()
    dataset.imag.plot(cmap="jet", data_only=True, clear=False)
    show()


# apodization


def test_nmr_2D_em_x(NMR_dataset_2D):
    dataset = NMR_dataset_2D.copy()
    assert dataset.shape == (96, 948)
    dataset.plot_map()  # plot original

    dataset = NMR_dataset_2D.copy()
    dataset.plot_map()
    dataset.em(lb=50.0 * ur.Hz, axis=-1)
    assert dataset.shape == (96, 948)
    dataset.plot_map(cmap="copper", data_only=True, clear=False)  # em on dim=x

    dataset = NMR_dataset_2D.copy()
    dataset.plot_map()
    dataset.em(lb=50.0 * ur.Hz, dim="x")
    assert dataset.shape == (96, 948)
    dataset.plot_map(cmap="copper", data_only=True, clear=False)  # em on dim=x

    show()


def test_nmr_2D_em_y(NMR_dataset_2D):
    dataset = NMR_dataset_2D.copy()
    assert dataset.shape == (96, 948)
    dataset.plot_map()  # plot original

    dataset = NMR_dataset_2D.copy()
    dataset.plot_map()
    dataset.em(lb=50.0 * ur.Hz, dim=0)
    assert dataset.shape == (96, 948)
    dataset.plot_map(cmap="copper", data_only=True, clear=False)  # em on dim=x

    dataset = NMR_dataset_2D.copy()
    dataset.plot_map()
    dataset.em(lb=50.0 * ur.Hz, dim="y")
    assert dataset.shape == (96, 948)
    dataset.plot_map(cmap="copper", data_only=True, clear=False)  # em on dim=x

    show()


def test_nmr_2D(NMR_dataset_2D):
    dataset = NMR_dataset_2D
    dataset.plot(nlevels=20)  # , start=0.15)
    show()
