# -*- coding: utf-8 -*-
# flake8: noqa


import pytest
import numpy as np

from spectrochempy.utils import show

pytestmark = pytest.mark.skip("WIP with NMR data")


def test_smooth(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()  # normalize
    dataset = dataset.fft(tdeff=8192, size=2 ** 15) + np.random.random(2 ** 15) * 5.0
    dataset.plot()

    s = dataset.smooth()
    s.plot(clear=False, color="r", xlim=[20, -20])

    show()


def test_smooth_2D(IR_dataset_2D):
    dataset = IR_dataset_2D.copy()
    dataset /= dataset.real.data.max()  # nromalize
    dataset += np.random.random(dataset.shape[-1]) * 0.02

    s = dataset.smooth(length=21)
    (dataset + 0.25).plot(xlim=[4000, 3000])
    s.plot(cmap="copper", clear=False, xlim=[4000, 3000])

    s2 = s.smooth(length=21, dim="y")
    (s2 - 0.25).plot(cmap="jet", clear=False, xlim=[4000, 3000])

    show()
