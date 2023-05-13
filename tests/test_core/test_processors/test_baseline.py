# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import os

import pytest

import spectrochempy as scp
from spectrochempy import NDDataset
from spectrochempy.analysis.preprocessing.baseline import Baseline
from spectrochempy.core.units import ur

# noinspection PyUnresolvedReferences
from spectrochempy.utils.plots import show
from spectrochempy.utils.testing import (
    assert_dataset_almost_equal,
    assert_dataset_equal,
)

path = os.path.dirname(os.path.abspath(__file__))


def test_preprocessing_baseline(IR_dataset_2D):

    # define a 1D test dataset (1 spectrum)
    dataset = IR_dataset_2D[10].squeeze()
    dataset[:, 1290.0:890.0] = scp.MASKED
    # minimal process
    basc1 = Baseline()
    basc1.fit(dataset)
    corr = basc1.transform()
    baseline = basc1.baseline
    assert baseline.shape == dataset.shape
    dataset.plot()
    corr.plot(clear=False, color="g")
    baseline.plot(clear=False, color="r")
    scp.show()

    # als process
    basc1 = Baseline(log_level="INFO")
    basc1.model = "als"
    basc1.mu = 0.5 * 10**9
    basc1.asymmetry = 0.001
    basc1.fit(dataset)
    corr = basc1.transform()
    baseline = basc1.baseline
    assert baseline.shape == dataset.shape
    dataset.plot()
    corr.plot(clear=False, color="g")
    baseline.plot(clear=False, color="r")
    scp.show()

    # with mask on some wavenumbers
    dataset[882.0:1280.0] = scp.MASKED
    basc2 = Baseline()
    basc2.fit(dataset)
    assert basc2.baseline.shape == dataset.shape

    # define a 2D test dataset (6 spectra)
    dataset = IR_dataset_2D[::10]

    # minimal process
    basc3 = Baseline()
    basc3.fit(dataset)
    assert basc3.baseline.shape == dataset.shape

    # now define ranges and interpolation=pchip
    basc3.ranges = [[6000.0, 3500.0], [2200.0, 1500.0]]
    basc3.model = "pchip"

    # and fit again (for example, taking only the second spectra)
    basc3.fit(dataset[1])

    # change the interpolation method
    basc3.model = "polynomial"
    basc3.order = 3
    basc3.fit(dataset)

    # multivariate
    basc3.multivariate = True
    basc3.model = "pchip"
    basc3.n_components = 5

    dataset = IR_dataset_2D
    dataset[:, 1290.0:890.0] = scp.MASKED
    basc3.ranges = (
        [5900.0, 5400.0],
        4550.0,
        [4500.0, 4000.0],
        [2100.0, 2000.0],
        [1550.0, 1555.0],
    )
    basc3.fit(dataset)

    basc3.model = "polynomial"
    basc3.order = 6
    basc3.fit(dataset)

    basc3.baseline[::10].plot(cmap=None, color="r")
    dataset[::10].plot(clear=False)
    scp.show()

    basc3.transform().plot()
    scp.show()

    # MS profiles, we want to make a baseline correction
    # on the ion current vs. time axis:
    ms = scp.read("msdata/ion_currents.asc", timestamp=False)
    msT = ms[4000.0:9000.0, :].T
    blc = scp.Baseline()
    # blc.ranges = [[10.0, 11.0], [19.0, 20.0]]
    blc.model = "polynomial"
    blc.order = 1
    blc.fit(msT)
    blc.corrected.plot()
    scp.show()


@pytest.mark.skip()
def test_ab_nmr(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()  # nromalize

    dataset.em(10.0 * ur.Hz, inplace=True)
    dataset = dataset.fft(tdeff=8192, size=2**15)
    dataset = dataset[150.0:-150.0] + 1.0

    dataset.plot()

    transf = dataset.copy()
    transfab = transf.ab(window=0.25)
    transfab.plot(clear=False, color="r")

    transf = dataset.copy()
    base = transf.ab(mode="poly", dryrun=True)
    transfab = transf - base
    transfab.plot(xlim=(150, -150), clear=False, color="b")
    base.plot(xlim=(150, -150), ylim=[-2, 10], clear=False, color="y")

    show()
