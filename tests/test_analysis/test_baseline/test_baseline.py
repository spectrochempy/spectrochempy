# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import os

import pytest

import spectrochempy as scp
from spectrochempy.core.units import ur
from spectrochempy.processing.baselineprocessing.baselineprocessing import Baseline
from spectrochempy.utils.plots import show
from spectrochempy.utils.testing import assert_dataset_equal

path = os.path.dirname(os.path.abspath(__file__))


def test_baseline(IR_dataset_2D):
    # define a 1D test dataset (1 spectrum)
    dataset = IR_dataset_2D[10].squeeze()
    # minimal process
    blc = Baseline()
    blc.fit(dataset)
    corr = blc.transform()
    baseline = blc.baseline
    assert baseline.shape == dataset.shape
    dataset.plot()
    corr.plot(clear=False, color="g")
    baseline.plot(clear=False, color="r")

    # asls process
    blc = Baseline(log_level="INFO")
    blc.model = "asls"
    blc.mu = 0.5 * 10**9
    blc.asymmetry = 0.001
    blc.fit(dataset)
    corr = blc.transform()
    baseline = blc.baseline
    assert baseline.shape == dataset.shape
    dataset.plot()
    corr.plot(clear=False, color="g")
    baseline.plot(clear=False, color="r")

    # with mask on some wavenumbers
    dataset[882.0:1280.0] = scp.MASKED
    blc = Baseline()
    blc.fit(dataset)
    corr = blc.transform()
    baseline = blc.baseline
    assert baseline.shape == dataset.shape
    dataset.plot()
    corr.plot(clear=False, color="g")
    baseline.plot(clear=False, color="r")

    # asls process with mask
    blc = Baseline(log_level="INFO")
    blc.model = "asls"
    blc.mu = 0.5 * 10**9
    blc.asymmetry = 0.001
    blc.fit(dataset)
    corr = blc.transform()
    baseline = blc.baseline
    assert baseline.shape == dataset.shape
    dataset.plot()
    corr.plot(clear=False, color="g")
    baseline.plot(clear=False, color="r")

    # define a 2D test dataset (6 spectra)
    dataset = IR_dataset_2D[::10]

    # minimal process
    basc3 = Baseline()
    basc3.fit(dataset)
    assert basc3.baseline.shape == dataset.shape

    # now define ranges and interpolation=pchip
    basc3.ranges = [[6000.0, 3500.0], [2200.0, 1500.0]]
    basc3.model = "polynomial"
    basc3.order = "pchip"

    # and fit again (for example, taking only the second spectra)
    basc3.fit(dataset[1])

    # change the interpolation method
    basc3.model = "polynomial"
    basc3.order = 3
    basc3.fit(dataset)

    # multivariate
    basc3.multivariate = True
    basc3.model = "polynomial"
    basc3.order = "pchip"
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

    basc3.transform().plot()

    # nmf multivariate
    basc3.multivariate = "nmf"
    basc3.model = "polynomial"
    basc3.order = 6
    basc3.n_components = 5
    basc3.fit(dataset)

    basc3.baseline[::10].plot(cmap=None, color="r")
    dataset[::10].plot(clear=False)

    basc3.transform().plot()

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


def test_baseline_sequential_asls(IR_dataset_2D):
    # test AsLS in sequential mode on a 2D spectra dataset

    blc = scp.Baseline(
        log_level="INFO",
    )

    blc.multivariate = False  # use a sequential baseline correction approach
    blc.model = "asls"  # use a asls model
    blc.mu = 10**8  # set the regularization parameter mu (smoothness)
    blc.asymmetry = 0.002

    ndp = IR_dataset_2D[::5]
    ndp[:, 1290.0:890.0] = scp.MASKED

    blc.fit(ndp)

    baseline = blc.baseline
    corrected = blc.corrected

    _ = corrected[0].plot()
    _ = baseline[0].plot(clear=False, color="red", ls="-")
    _ = ndp[0].plot(clear=False, color="green", ls="--")

    _ = corrected[-1].plot()
    _ = baseline[-1].plot(clear=False, color="red", ls="-")
    _ = ndp[10].plot(clear=False, color="green", ls="--")

    _ = corrected.plot()

    scp.show()

    # it works but not very well adapted to a situation where the regularization
    # parameter mu and may be asymmetry should be adapted to each spectra.


def test_preprocessing_nddataset_methods(IR_dataset_2D):
    ndp = IR_dataset_2D[::5]
    ndp[:, 1290.0:890.0] = scp.MASKED

    # baseline
    baseline = ndp.get_baseline()
    assert baseline.shape == ndp.shape
    baseline = ndp.get_baseline(model="asls", lamb=10**8, asymmetry=0.002)
    assert baseline.shape == ndp.shape

    # asls
    ndpcor = scp.asls(ndp, lamb=10**8, asymmetry=0.002)
    assert_dataset_equal(ndpcor, ndp - baseline)

    # snip
    ndpcor = scp.snip(ndp, snip_width=150)
    baseline = ndp.get_baseline(model="snip", snip_width=150)
    assert_dataset_equal(ndpcor, ndp - baseline)


def test_baseline_polynomial(IR_dataset_2D):
    X = IR_dataset_2D[::5]

    blc = scp.Baseline()
    blc.model = "polynomial"
    blc.order = 2
    blc.ranges = [[4000.0, 4001.0], [2000.0, 2001.0]]
    blc.fit(X)

    blc.plot()
