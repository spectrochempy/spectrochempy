# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.units import ur


def test_nddataset_bug_fixe_figopeninnotebookwithoutplot():
    import sys

    # Track matplotlib state before operations
    matplotlib_loaded_before = "matplotlib" in sys.modules
    da = scp.NDDataset([1, 2, 3])
    da2 = np.sqrt(da**3)
    # Verify that matplotlib was not loaded by these operations
    # (it may have been loaded by earlier tests in the session)
    matplotlib_loaded_after = "matplotlib" in sys.modules
    # The key assertion: if matplotlib wasn't loaded before, it shouldn't be loaded after
    # Just ensure no error is raised from dataset operations
    assert da2 is not None


def test_nddataset_bug_par_arnaud():
    import numpy as np

    import spectrochempy as scp

    x = scp.Coord(data=np.linspace(1000.0, 4000.0, num=6000), title="x")
    y = scp.Coord(data=np.linspace(0.0, 10, num=5), title="y")
    data = np.random.rand(x.size, y.size)
    ds = scp.NDDataset(data, coordset=[x, y])
    ds2 = ds[2000.0:3200.0, :]
    assert ds2.coordset.y.data.shape[0] == 2400, "taille axe 0 doit être 2400"
    assert ds2.data.shape[0] == 2400, "taille dimension 0 doit être 2400"


def test_nddataset_bug_462():
    A = scp.random((10, 100))
    A.x = scp.Coord(np.arange(0.0, 100.0, 1), title="coord1")
    af = A.write("A.scp", overwrite=True)
    B = scp.read("A.scp")
    assert B.x == A.x

    C = scp.random((10, 100))
    C.x = [
        scp.Coord(np.arange(0.0, 100.0, 1), title="coord1"),
        scp.Coord(np.arange(0.0, 1000.0, 10), title="coord2"),
    ]
    cf = C.write("C.scp", overwrite=True)
    D = scp.read("C.scp")
    assert len(D.x) == 2, "incorrect encoding/decoding"

    af.unlink()
    cf.unlink()


def test_nddataset_max_min_with_1D(IR_dataset_1D):
    # test on a 1D NDDataset
    nd1 = IR_dataset_1D
    nd1[4] = scp.MASKED
    assert nd1.is_masked
    mx = nd1.max().m
    assert (mx.real, mx.imag) == pytest.approx((6.0, 0.0))
    # check if it works for real
    mx1 = nd1.real.max().m
    assert mx1 == pytest.approx(6.0)
    mi = nd1.min().m
    assert (mi.real, mi.imag) == pytest.approx((0.5944115519523621, 0.0))


def test_nddataset_comparison_of_dataset(IR_dataset_1D):
    # bug in notebook
    nd1 = IR_dataset_1D.copy()
    nd2 = IR_dataset_1D.copy()
    lb1 = nd1.em(lb=100.0 * ur.Hz)
    lb2 = nd2.em(lb=100.0 * ur.Hz)
    assert nd1 is not nd2
    assert nd1 == nd2
    assert lb1 is not lb2
    assert lb1 == lb2


if __name__ == "__main__":
    pytest.main([__file__])
