# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for sklearn-compatible API on analysis estimators."""

import numpy as np
import pytest

from spectrochempy import NDDataset
from spectrochempy.analysis.decomposition.pca import PCA
from spectrochempy.processing.baselineprocessing.baselineprocessing import Baseline
from spectrochempy.utils.exceptions import SpectroChemPyError


class TestAnalysisGetParams:
    def test_pca_get_params(self):
        pca = PCA(n_components=5)
        params = pca.get_params()
        assert isinstance(params, dict)
        assert params["n_components"] == 5
        assert "svd_solver" in params

    def test_baseline_get_params(self):
        bl = Baseline()
        params = bl.get_params()
        assert isinstance(params, dict)
        assert "multivariate" in params

    def test_params_reflect_current_values(self):
        pca = PCA()
        pca.n_components = 3
        params = pca.get_params()
        assert params["n_components"] == 3


class TestAnalysisSetParams:
    def test_pca_set_params(self):
        pca = PCA()
        result = pca.set_params(n_components=7)
        assert result is pca
        assert pca.n_components == 7

    def test_set_params_chaining(self):
        pca = PCA()
        pca.set_params(n_components=2).fit(
            NDDataset(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        )
        assert pca.n_components == 2

    def test_set_params_invalid_raises(self):
        pca = PCA()
        with pytest.raises(SpectroChemPyError, match="Invalid parameter"):
            pca.set_params(nonexistent_param=42)


class TestAnalysisRepr:
    def test_pca_repr(self):
        pca = PCA(n_components=5)
        r = repr(pca)
        assert "PCA" in r
        assert "n_components=5" in r

    def test_baseline_repr(self):
        bl = Baseline()
        r = repr(bl)
        assert "Baseline" in r


class TestAnalysisSklearnClone:
    def test_sklearn_clone_not_guaranteed_for_traitlets_classes(self):
        """
        sklearn.base.clone() may fail on AnalysisConfigurable subclasses
        because traitlets modifies constructor arguments (e.g., empty
        lists become internal traitlet structures).  get_params and
        set_params work, but clone compatibility is best-effort only.
        """
        try:
            from sklearn.base import clone
        except ImportError:
            pytest.skip("scikit-learn not installed")

        pca = PCA(n_components=3)
        # clone may raise RuntimeError; that is acceptable
        try:
            cloned = clone(pca)
        except RuntimeError as exc:
            assert "Cannot clone" in str(exc)
            pytest.xfail("traitlets traits break sklearn clone identity check")
