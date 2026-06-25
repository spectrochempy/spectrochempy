# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the SVD class
"""

from os import environ

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis.decomposition.svd import SVD
from spectrochempy.utils import docutils as chd


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_SVD_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.svd"
    chd.check_docstrings(
        module,
        obj=scp.SVD,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "EX02", "ES01", "GL11", "GL08", "PR01"],
    )


@pytest.fixture()
def low_rank_dataset():
    y = scp.Coord.arange(4, title="sample")
    x = scp.Coord.arange(5, title="feature")
    data = np.zeros((4, 5))
    data[0, 0] = 5.0
    data[1, 1] = 3.0
    return scp.NDDataset(
        data,
        coordset=[y, x],
        units="absorbance",
        title="synthetic low-rank matrix",
    )


def test_svd(low_rank_dataset):
    dataset = low_rank_dataset
    svd = SVD()
    result = svd.fit(dataset)

    assert result is svd
    assert svd.U.shape == (4, 4)
    assert svd.VT.shape == (4, 5)
    assert svd.sv.shape == (4,)
    assert svd.sv.title == "Singular values"
    assert svd.sv.dims == ["k"]
    assert_allclose(svd.s, [5.0, 3.0, 0.0, 0.0])
    assert_allclose(svd.ev_ratio.data, [2500.0 / 34.0, 900.0 / 34.0, 0.0, 0.0])
    assert_allclose(svd.ev_cum.data, [2500.0 / 34.0, 100.0, 100.0, 100.0])

    # Fully masked rows/columns are ignored by the SVD calculation while the
    # public input dataset keeps its original shape.
    masked = dataset.copy()
    masked[:, 4] = scp.MASKED
    masked[3] = scp.MASKED

    svd.fit(masked)
    assert svd.X.shape == dataset.shape
    assert svd.U.shape == (3, 3)
    assert svd.VT.shape == (3, 4)
    assert_allclose(svd.s, [5.0, 3.0, 0.0])
    assert_allclose(svd.ev_ratio.data, [2500.0 / 34.0, 900.0 / 34.0, 0.0, 0.0])

    svd.full_matrices = True
    svd.fit(masked)

    assert svd.U.shape == (3, 3)
    assert svd.VT.shape == (4, 4)
    assert_allclose(svd.s, [5.0, 3.0, 0.0])


def test_svd_compute_uv_false(low_rank_dataset):
    dataset = low_rank_dataset
    svd = SVD(compute_uv=False)
    result = svd.fit(dataset)

    assert result is svd
    # _outfit is always a (U, s, VT) tuple after _fit normalisation
    assert isinstance(svd._outfit, tuple)
    assert len(svd._outfit) == 3
    # U and VT are None when compute_uv=False
    assert svd.U is None
    assert svd.VT is None
    # Singular values are correct
    assert_allclose(svd.s, [5.0, 3.0, 0.0, 0.0])
    assert svd.sv.shape == (4,)
    assert svd.sv.title == "Singular values"
    # Diagnostics work correctly
    assert svd.ev.shape == (4,)
    assert svd.ev_ratio.shape == (4,)
    assert svd.ev_cum.shape == (4,)
    assert_allclose(svd.ev_ratio.data, [2500.0 / 34.0, 900.0 / 34.0, 0.0, 0.0])
    assert_allclose(svd.ev_cum.data, [2500.0 / 34.0, 100.0, 100.0, 100.0])
    # repr shows U and VT not computed
    assert "U and VT not computed" in repr(svd)

    # full_matrices=True + compute_uv=False
    svd2 = SVD(full_matrices=True, compute_uv=False)
    svd2.fit(dataset)
    assert svd2.U is None
    assert svd2.VT is None
    assert_allclose(svd2.s, [5.0, 3.0, 0.0, 0.0])
