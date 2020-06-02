# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from spectrochempy import *
import pytest

pytestmark = pytest.mark.skip("WIP with NMR data")

def test_smooth(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()  # nromalize
    dataset = dataset.fft(tdeff=8192, size=2 ** 15) + np.random.random(2 ** 15) * 5.
    dataset.plot()

    s = dataset.smooth(window='hanning')
    s.plot(clear=False, color='r', xlim=[20, -20])

    show()


def test_smooth_2D(IR_dataset_2D):
    dataset = IR_dataset_2D.copy()
    dataset /= dataset.real.data.max()  # nromalize
    dataset += np.random.random(dataset.shape[-1]) * 0.02

    s = dataset.smooth(window_length=21)
    (dataset + 0.25).plot(xlim=[4000, 3000])
    s.plot(cmap='copper', clear=False, xlim=[4000, 3000])

    s2 = s.smooth(window_length=21, dim='y')
    (s2 - 0.25).plot(cmap='jet', clear=False, xlim=[4000, 3000])

    t = dataset.savgol_filter(window_length=21, polyorder=2)
    t.plot(cmap='copper', clear=True, xlim=[4000, 3000])

    t2 = t.savgol_filter(window_length=21, polyorder=0, dim='y')
    (t2 - 0.25).plot(cmap='jet', clear=False, xlim=[4000, 3000])

    show()



