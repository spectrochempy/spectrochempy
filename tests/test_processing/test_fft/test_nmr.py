# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os

import numpy as np

from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import (
    assert_array_almost_equal,
    assert_array_equal,
)


# ---------------------------------------------------------------------------
# 1D reader
# ---------------------------------------------------------------------------


def test_nmr_reader_1D():
    import spectrochempy as scp

    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_1d"
    )
    ndd = scp.read_topspin(path, expno=1, remove_digital_filter=True)
    assert ndd.__str__() == "NDDataset: [complex128] count (size: 12411)"
    assert "coordinates" in ndd._repr_html_()
    assert ndd.shape == (12411,)
    assert ndd.data.dtype == np.complex128


# ---------------------------------------------------------------------------
# 1D apodization
# ---------------------------------------------------------------------------


def test_nmr_1D_em(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()  # normalize
    original = dataset.data.copy()

    # em with inplace=False returns a new object with modified data
    arr, apod = dataset.em(lb=100, retapod=True)
    assert not np.array_equal(arr.data, original)
    assert arr.shape == dataset.shape
    assert_array_almost_equal(apod[1], 0.9987, decimal=4)

    # em with inplace=True modifies the dataset in place
    dataset2 = NMR_dataset_1D.copy()
    dataset2 /= dataset2.real.data.max()
    original2 = dataset2.data.copy()
    dataset2.em(lb=100.0 * ur.Hz, inplace=True)
    assert not np.array_equal(dataset2.data, original2)


def test_nmr_1D_gm(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset /= dataset.real.data.max()

    dataset_gm, apod = dataset.gm(lb=-100.0 * ur.Hz, gb=100.0 * ur.Hz, retapod=True)
    assert dataset_gm.shape == dataset.shape
    assert not np.array_equal(dataset_gm.data, dataset.data)
    assert apod.shape == dataset.shape


# ---------------------------------------------------------------------------
# 1D FFT
# ---------------------------------------------------------------------------


def test_nmr_fft_1D(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize
    dataset1D.x.ito("s")

    # FFT produces correct shape and dtype
    new = dataset1D.fft(tdeff=8192, size=2**15)
    assert new.shape == (2**15,)
    assert new.data.dtype == np.complex128
    assert np.any(new.data.imag != 0)  # result should be complex

    # IFFT preserves energy (Parseval's theorem)
    energy_time = np.sum(np.abs(dataset1D.data) ** 2)
    new2 = new.ifft()
    energy_roundtrip = np.sum(np.abs(new2.data) ** 2)
    assert abs(energy_roundtrip / energy_time - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 1D manual phasing
# ---------------------------------------------------------------------------


def test_nmr_manual_1D_phasing(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize

    dataset1D.em(lb=10.0 * ur.Hz)
    transf = dataset1D.fft(tdeff=8192, size=2**15)

    # phasing with default pivot and phc0=0 should return same data
    transfph = transf.pk(verbose=True)
    assert_array_equal(transfph.data, transf.data)

    # with phc0=0 (default), pivot position doesn't change the data
    transfph3 = transf.pk(pivot=50, verbose=True)
    assert_array_equal(transfph3.data, transfph.data)

    # phc0 != 0 changes the data
    transfph4 = transf.pk(pivot=100, phc0=40.0, verbose=True)
    assert not np.array_equal(transfph4.data, transf.data)

    # inplace=True returns the same object
    transfph5 = transf.pk(pivot=100, verbose=True, inplace=True)
    assert transfph5 is transf


# ---------------------------------------------------------------------------
# 2D reader
# ---------------------------------------------------------------------------


def test_nmr_reader_2D():
    import spectrochempy as scp

    path = os.path.join(
        prefs.datadir, "nmrdata", "bruker", "tests", "nmr", "topspin_2d"
    )
    ndd = scp.read_topspin(path, expno=1, remove_digital_filter=True)
    assert "count" in ndd.__str__()
    assert "(shape: (y:96, x:474))" in ndd.__str__()
    assert "coordinates" in ndd._repr_html_()
    assert ndd.shape == (96, 474)


# ---------------------------------------------------------------------------
# 2D em (axis-specific, shape preservation)
# ---------------------------------------------------------------------------


def test_nmr_2D_em_x(NMR_dataset_2D):
    dataset = NMR_dataset_2D.copy()
    assert dataset.shape == (96, 474)

    # em on F2 axis preserves shape
    dataset.em(lb=50.0 * ur.Hz, axis=-1)
    assert dataset.shape == (96, 474)

    # em with dim="x" preserves shape
    dataset2 = NMR_dataset_2D.copy()
    dataset2.em(lb=50.0 * ur.Hz, dim="x")
    assert dataset2.shape == (96, 474)


def test_nmr_2D_em_y(NMR_dataset_2D):
    dataset = NMR_dataset_2D.copy()
    assert dataset.shape == (96, 474)

    # em on F1 axis preserves shape
    dataset.em(lb=50.0 * ur.Hz, dim=0)
    assert dataset.shape == (96, 474)

    # em with dim="y" preserves shape
    dataset2 = NMR_dataset_2D.copy()
    dataset2.em(lb=50.0 * ur.Hz, dim="y")
    assert dataset2.shape == (96, 474)
