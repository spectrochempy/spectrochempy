# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the SVD class
"""

from os import environ

import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis.decomposition.svd import SVD
from spectrochempy.utils import docstrings as chd
from spectrochempy.utils.constants import MASKED


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
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


# test svd
# -----------
def test_svd(IR_dataset_2D):
    dataset = IR_dataset_2D

    svd = SVD()
    svd.fit(dataset)

    assert_allclose(svd.ev_ratio[0].data, 94.539, rtol=1e-5, atol=0.0001)

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing
    dataset[10:12] = MASKED

    dataset.plot_stack()

    svd.fit(dataset)
    assert_allclose(svd.ev_ratio.data[0], 93.8, rtol=1e-4, atol=0.01)

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing
    dataset[10:12] = MASKED
    dataset.plot_stack()

    svd.full_matrices = True
    svd.fit(dataset)

    assert_allclose(svd.ev_ratio.data[0], 93.8, rtol=1e-4, atol=0.01)
