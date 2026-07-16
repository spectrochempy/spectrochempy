# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

import numpy as np
import pytest

import spectrochempy as scp

DATADIR = scp.preferences.datadir
NMRDATA = DATADIR / "nmrdata"
nmrdir = NMRDATA / "bruker" / "tests" / "nmr"


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_smooth_preserves_shape():
    path = nmrdir / "topspin_1d" / "1" / "fid"
    ndd = scp.read_topspin(path, expno=1, remove_digital_filter=True)
    dataset = ndd.copy()
    dataset /= dataset.real.data.max()
    dataset = dataset.fft(tdeff=8192, size=2**15)

    s = dataset.smooth(size=21)
    assert s.shape == dataset.shape


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_smooth_reduces_noise():
    path = nmrdir / "topspin_1d" / "1" / "fid"
    ndd = scp.read_topspin(path, expno=1, remove_digital_filter=True)
    dataset = ndd.copy()
    dataset /= dataset.real.data.max()
    dataset = dataset.fft(tdeff=8192, size=2**15) + np.random.random(2**15) * 5.0

    s = dataset.smooth(size=21)
    assert not np.array_equal(s.data, dataset.data)
    # smooth should reduce high-frequency noise in quiet regions
    assert np.std(np.abs(s.data[1000:2000])) < np.std(np.abs(dataset.data[1000:2000]))
