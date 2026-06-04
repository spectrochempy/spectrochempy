# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.decomposition.simplisma import SIMPLISMA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import docutils as chd
from spectrochempy.utils import testing


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    os.environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_SIMPLISMA_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.simplisma"
    chd.check_docstrings(
        module,
        obj=scp.SIMPLISMA,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


def test_simplisma_fit_returns_estimator(simplisma_dataset):
    sma = SIMPLISMA(n_components=2, log_level="WARNING")
    result = sma.fit(simplisma_dataset)
    assert result is sma


def test_simplisma_fit_shapes(simplisma_dataset):
    sma = SIMPLISMA(n_components=2, log_level="WARNING")
    sma.fit(simplisma_dataset)

    assert sma.C.shape == (20, 2)
    assert sma.St.shape == (2, 100)
    assert sma.Pt.shape == (2, 100)
    assert sma.s.shape == (2, 100)
    assert sma.n_components == 2

    testing.assert_dataset_equal(sma.X, simplisma_dataset)


def test_simplisma_finite_outputs(simplisma_dataset):
    sma = SIMPLISMA(n_components=2, log_level="WARNING")
    sma.fit(simplisma_dataset)

    assert np.all(np.isfinite(sma.C.data))
    assert np.all(np.isfinite(sma.St.data))
    assert np.all(np.isfinite(sma.Pt.data))
    assert np.all(np.isfinite(sma.s.data))


def test_simplisma_pure_variable_selection(simplisma_dataset):
    sma = SIMPLISMA(n_components=2, log_level="WARNING")
    sma.fit(simplisma_dataset)

    pure_idx = np.argmax(sma.Pt.data, axis=1)

    w = simplisma_dataset.x.data
    expected_1 = np.argmin(np.abs(w - 30.0))
    expected_2 = np.argmin(np.abs(w - 70.0))

    assert any(abs(idx - expected_1) <= 2 for idx in pure_idx)
    assert any(abs(idx - expected_2) <= 2 for idx in pure_idx)


def test_simplisma_transform(simplisma_dataset):
    sma = SIMPLISMA(n_components=2, log_level="WARNING")
    sma.fit(simplisma_dataset)

    C_from_transform = sma.transform()
    assert C_from_transform.shape == (20, 2)
    assert np.allclose(C_from_transform.data, sma.C.data)

    C_from_transform_with_X = sma.transform(simplisma_dataset)
    assert np.allclose(C_from_transform_with_X.data, sma.C.data)


def test_simplisma_inverse_transform(simplisma_dataset):
    sma = SIMPLISMA(n_components=2, log_level="WARNING")
    sma.fit(simplisma_dataset)

    X_recon = sma.inverse_transform()
    assert X_recon.shape == simplisma_dataset.shape
    assert np.all(np.isfinite(X_recon.data))
    assert X_recon.units == simplisma_dataset.units


def test_simplisma_n_components_validation(simplisma_dataset):
    with pytest.raises(ValueError, match="larger than 2"):
        SIMPLISMA(n_components=1).fit(simplisma_dataset)


def test_simplisma_3d_raises():
    data_3d = np.arange(60.0).reshape(3, 4, 5)
    ds_3d = NDDataset(data_3d)

    with pytest.raises(ValueError, match="only handles 2D"):
        SIMPLISMA(n_components=2).fit(ds_3d)


def test_simplisma_negative_warning(simplisma_dataset):
    ds_neg = simplisma_dataset.copy()
    ds_neg.data -= 0.5

    with pytest.warns(UserWarning, match="does not handle easily negative"):
        SIMPLISMA(n_components=2).fit(ds_neg)
