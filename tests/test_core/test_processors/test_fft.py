# -*- coding: utf-8 -*-
# flake8: noqa

import pytest

from spectrochempy.core import preferences as prefs
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_raises,
)


def test_nmr_fft_1D(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize
    dataset1D.x.ito("s")
    new = dataset1D.fft(tdeff=8192, size=2 ** 15)
    new2 = new.ifft()


def test_nmr_fft_1D_our_Hz(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize
    LB = 10.0 * ur.Hz
    GB = 50.0 * ur.Hz
    dataset1D.gm(gb=GB, lb=LB)
    new = dataset1D.fft(size=32000, ppm=False)
