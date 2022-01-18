# -*- coding: utf-8 -*-
# flake8: noqa


import os

import numpy as np
import pytest

from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur

from spectrochempy.utils import show
from spectrochempy.utils.testing import (
    assert_equal,
    assert_array_equal,
    assert_array_almost_equal,
)

# pytestmark = pytest.mark.skip("all tests still WIP")

DATADIR = prefs.datadir
NMRDATA = DATADIR / "nmrdata"

nmrdir = NMRDATA / "bruker" / "tests" / "nmr"


@pytest.mark.skipif(
    not NMRDATA.exists(),
    reason="Experimental data not available for testing",
)
def test_nmr_manual_1D_phasing(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize

    # dataset1D.em(10.0 * ur.Hz)  # inplace broadening
    # this do not work because wrapper do not accept positional argument:
    # TODO: modify this behaviour

    dataset1D.em(lb=10.0 * ur.Hz)
    transf = dataset1D.fft(tdeff=8192, size=2 ** 15)  # fft
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


@pytest.mark.skip("WIP")
def test_nmr_auto_1D_phasing():
    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_1d"
    )
    ndd = NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(lb=10.0 * ur.Hz, inplace=True)
    transf = ndd.fft(tdeff=8192, size=2 ** 15)
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


@pytest.mark.skip("WIP")
def test_nmr_multiple_manual_1D_phasing():
    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_1d"
    )
    ndd = NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(lb=10.0 * ur.Hz)  # inplace broadening

    transf = ndd.fft(tdeff=8192, size=2 ** 15)

    transfph1 = transf.pk(verbose=True)
    transfph1.plot(xlim=(20, -20), color="k")

    transfph2 = transf.pk(verbose=True)
    transfph2.plot(xlim=(20, -20), clear=False, color="r")

    transfph3 = transf.pk(52.43836, -16.8366, verbose=True)
    transfph3.plot(xlim=(20, -20), clear=False, color="b")

    show()


@pytest.mark.skip("WIP")
def test_nmr_multiple_auto_1D_phasing():
    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_1d"
    )
    ndd = NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(lb=10.0 * ur.Hz)  # inplace broadening

    transf = ndd.fft(tdeff=8192, size=2 ** 15)
    transf.plot(xlim=(20, -20), ls=":", color="k")

    t1 = transf.apk(algorithm="neg_peak", verbose=True)
    (t1 - 5.0).plot(xlim=(20, -20), clear=False, color="b")

    t2 = t1.apk(algorithm="neg_area", verbose=True)
    (t2 - 10).plot(xlim=(20, -20), clear=False, ls="-.", color="m")

    t3 = t2.apk(algorithm="acme", verbose=True)
    (t3 - 15).plot(xlim=(20, -20), clear=False, color="r")

    show()
