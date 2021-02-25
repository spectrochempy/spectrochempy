# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =
# ======================================================================================================================

import os

import pytest

# noinspection PyUnresolvedReferences
import spectrochempy as scp
from spectrochempy import (show, BaselineCorrection, NDDataset, ur, )
from spectrochempy.utils.testing import assert_dataset_almost_equal

path = os.path.dirname(os.path.abspath(__file__))


# @pytest.mark.skip("erratic failing!")
def test_basecor_sequential(IR_dataset_2D):
    dataset = IR_dataset_2D[5]
    basc = BaselineCorrection(dataset)

    s = basc([6000., 3500.], [2200., 1500.], method='sequential', interpolation='pchip')
    s.plot()

    s1 = basc([6000., 3500.], [2200., 1500.], method='sequential', interpolation='polynomial')
    s1.plot(clear=False, color='red')

    dataset = IR_dataset_2D[5]  # with LinearCoord
    basc = BaselineCorrection(dataset)

    s2 = basc([6000., 3500.], [2200., 1500.], method='sequential', interpolation='pchip')
    assert_dataset_almost_equal(s, s2, decimal=5)
    s2.plot(clear=False, color='green')

    s3 = basc([6000., 3500.], [2200., 1500.], method='sequential', interpolation='polynomial')
    assert_dataset_almost_equal(s1, s3, decimal=5)
    s3.plot(clear=False, color='cyan')

    show()

    dataset = IR_dataset_2D[:15]
    basc = BaselineCorrection(dataset)
    s = basc([6000., 3500.], [2200., 1500.], method='sequential', interpolation='pchip')
    s.plot()
    s = basc([6000., 3500.], [2200., 1500.], method='sequential', interpolation='polynomial')
    s.plot(cmap='copper')

    show()


def test_basecor_multivariate(IR_dataset_2D):
    dataset = IR_dataset_2D[5]

    basc = BaselineCorrection(dataset)
    s = basc([6000., 3500.], [1800., 1500.], method='multivariate', interpolation='pchip')
    s.plot()
    s = basc([6000., 3500.], [1800., 1500.], method='multivariate', interpolation='polynomial')
    s.plot(clear=False, color='red')

    dataset = IR_dataset_2D[:15]

    basc = BaselineCorrection(dataset)
    s = basc([6000., 3500.], [1800., 1500.], method='multivariate', interpolation='pchip')
    s.plot()
    s = basc([6000., 3500.], [1800., 1500.], method='multivariate', interpolation='polynomial')
    s.plot(cmap='copper')
    show()


def test_notebook_basecor_bug():
    dataset = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))

    s = dataset[:, 1260.0:5999.0]
    s = s - s[-1]

    # Important note that we use floating point number
    # integer would mean points, not wavenumbers!

    basc = BaselineCorrection(s)

    ranges = [[1261.86, 1285.89], [1556.30, 1568.26], [1795.00, 1956.75], [3766.03, 3915.81], [4574.26, 4616.04],
              [4980.10, 4998.01], [5437.52, 5994.70]]  # predifined ranges

    _ = basc.run(*ranges, method='multivariate', interpolation='pchip', npc=5, figsize=(6, 6), zoompreview=4)

    # The regions used to set the baseline are accessible using the `ranges`
    #  attibute:
    ranges = basc.ranges
    print(ranges)

    basc.corrected.plot_stack()


def test_issue_227():
    # IR spectrum, we want to make a baseline correction on the absorbance vs. time axis:
    ir = scp.read('irdata/nh4y-activation.spg')
    ir.y = ir.y - ir[0].y
    irs = ir[:, 2000.0:2020.0]
    blc = scp.BaselineCorrection(irs, dim='y')
    blc.compute(*[[0., 2.e3], [3.0e4, 3.3e4]], interpolation='polynomial', order=1, method='sequential')
    blc.corrected.plot()
    scp.show()

    # MS profiles, we want to make a baseline correction on the ion current vs. time axis:
    ms = scp.read('msdata/ion_currents.asc', timestamp=False)
    blc = scp.BaselineCorrection(ms[10.:20., :], dim=0)
    blc.compute(*[[10., 11.], [19., 20.]], interpolation='polynomial', order=1, method='sequential')
    blc.corrected.T.plot()
    scp.show()


@pytest.mark.skip()
def test_ab_nmr(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()  # nromalize

    dataset.em(10. * ur.Hz, inplace=True)
    dataset = dataset.fft(tdeff=8192, size=2 ** 15)
    dataset = dataset[150.0:-150.] + 1.

    dataset.plot()

    transf = dataset.copy()
    transfab = transf.ab(window=.25)
    transfab.plot(clear=False, color='r')

    transf = dataset.copy()
    base = transf.ab(mode="poly", dryrun=True)
    transfab = transf - base
    transfab.plot(xlim=(150, -150), clear=False, color='b')
    base.plot(xlim=(150, -150), ylim=[-2, 10], clear=False, color='y')

    show()
