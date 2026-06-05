# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the PLSRegression module.

Uses deterministic synthetic latent-variable data instead of real datasets.
Numerical correctness is validated against sklearn's PLSRegression.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.cross_decomposition import PLSRegression as sklPLSRegression

import spectrochempy as scp
from spectrochempy.analysis.crossdecomposition.pls import PLSRegression
from spectrochempy.utils import docutils as chd
from spectrochempy.utils.constants import MASKED


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pls_data():
    """Deterministic synthetic latent-variable data for PLS regression testing.

    Generates X and Y from a low-rank structure: X = T @ P + noise,
    Y = T @ Q + noise, where T are latent scores, P and Q are loadings.

    Returns a dict with keys:
        Xc, Xv, Yc, Yv  : NDDataset (calibration and validation)
        yc, yv          : NDDataset, first column of Y (univariate response)
        n_components    : int
        n_cal, n_val    : int
        n_features      : int
        n_targets       : int
    """
    rng = np.random.default_rng(42)
    n_cal = 30
    n_val = 10
    n_features = 50
    n_components = 3
    n_targets = 2

    T_cal = rng.normal(0, 1, (n_cal, n_components))
    T_val = rng.normal(0, 1, (n_val, n_components))
    P = rng.normal(0, 1, (n_components, n_features))
    Q = rng.normal(0, 1, (n_components, n_targets))

    noise_X = 0.05 * rng.normal(0, 1, (n_cal, n_features))
    noise_Y = 0.05 * rng.normal(0, 1, (n_cal, n_targets))
    Xc_data = T_cal @ P + noise_X
    Yc_data = T_cal @ Q + noise_Y
    Xv_data = T_val @ P + 0.05 * rng.normal(0, 1, (n_val, n_features))
    Yv_data = T_val @ Q + 0.05 * rng.normal(0, 1, (n_val, n_targets))

    cal_coord = scp.Coord(np.arange(n_cal), title="samples")
    val_coord = scp.Coord(np.arange(n_cal, n_cal + n_val), title="samples")
    feat_coord = scp.Coord(np.arange(n_features), title="wavelength", units="nm")
    targ_coord = scp.Coord(np.arange(n_targets), title="properties")

    Xc = scp.NDDataset(
        Xc_data,
        coordset=[cal_coord, feat_coord],
        title="X calibration",
        units="absorbance",
    )
    Xv = scp.NDDataset(
        Xv_data,
        coordset=[val_coord, feat_coord],
        title="X validation",
        units="absorbance",
    )
    Yc = scp.NDDataset(
        Yc_data,
        coordset=[cal_coord, targ_coord],
        title="Y calibration",
    )
    Yv = scp.NDDataset(
        Yv_data,
        coordset=[val_coord, targ_coord],
        title="Y validation",
    )
    yc = Yc[:, 0]
    yv = Yv[:, 0]

    return {
        "Xc": Xc,
        "Xv": Xv,
        "Yc": Yc,
        "Yv": Yv,
        "yc": yc,
        "yv": yv,
        "n_components": n_components,
        "n_cal": n_cal,
        "n_val": n_val,
        "n_features": n_features,
        "n_targets": n_targets,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_PLS_docstrings():
    """Verify docstrings follow project conventions."""
    chd.PRIVATE_CLASSES = []
    module = "spectrochempy.analysis.crossdecomposition.pls"
    chd.check_docstrings(
        module,
        obj=scp.PLSRegression,
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01", "SS01"],
    )


class TestPLSUnivariate:
    """PLS regression with a single response variable."""

    @pytest.fixture(autouse=True)
    def _setup(self, pls_data):
        self.d = pls_data
        self.pls = PLSRegression(n_components=self.d["n_components"])
        self.pls.fit(self.d["Xc"], self.d["yc"])
        self.skl = sklPLSRegression(n_components=self.d["n_components"])
        self.skl.fit(self.d["Xc"].data, self.d["yc"].data.squeeze())

    def test_fit_attributes(self):
        pls = self.pls
        d = self.d
        assert pls._fitted
        assert pls.n_components == d["n_components"]
        assert pls.x_loadings.shape == (d["n_components"], d["n_features"])
        assert pls.y_loadings.shape == (d["n_components"], 1)
        assert pls.x_scores.shape == (d["n_cal"], d["n_components"])
        assert pls.y_scores.shape == (d["n_cal"], d["n_components"])
        assert np.all(np.isfinite(pls.x_loadings.data))
        assert np.all(np.isfinite(pls.y_loadings.data))
        assert np.all(np.isfinite(pls.x_scores.data))
        assert np.all(np.isfinite(pls.y_scores.data))

    def test_fit_against_sklearn(self):
        pls = self.pls
        skl = self.skl
        assert_allclose(pls.x_loadings.data, skl.x_loadings_.T)
        assert_allclose(pls.y_loadings.data.squeeze(), skl.y_loadings_.squeeze().T)
        assert_allclose(pls.x_scores.data, skl.x_scores_)
        assert_allclose(pls.y_scores.data, skl.y_scores_)

    def test_score(self):
        pls = self.pls
        skl = self.skl
        d = self.d
        assert_allclose(pls.score(), skl.score(d["Xc"].data, d["yc"].data.squeeze()))
        score_val = pls.score(d["Xv"], d["yv"])
        score_val_skl = skl.score(d["Xv"].data, d["yv"].data.squeeze())
        assert_allclose(score_val, score_val_skl, rtol=1e-5)
        assert 0.0 < score_val <= 1.0

    def test_predict(self):
        pls = self.pls
        skl = self.skl
        d = self.d
        y_hat = pls.predict(d["Xv"])
        assert isinstance(y_hat, scp.NDDataset)
        assert y_hat.shape == (d["n_val"], 1)
        y_hat_skl = skl.predict(d["Xv"].data)
        assert_allclose(y_hat.data.squeeze(), y_hat_skl.squeeze())
        assert np.all(np.isfinite(y_hat.data))

    def test_transform(self):
        pls = self.pls
        skl = self.skl
        d = self.d
        x_scores = pls.transform()
        assert isinstance(x_scores, scp.NDDataset)
        assert x_scores.shape == (d["n_cal"], d["n_components"])
        assert_allclose(x_scores.data, skl.transform(d["Xc"].data))

        x_scores_v = pls.transform(d["Xv"])
        assert x_scores_v.shape == (d["n_val"], d["n_components"])
        assert_allclose(x_scores_v.data, skl.transform(d["Xv"].data))

    def test_transform_with_y(self):
        pls = self.pls
        d = self.d
        result = pls.transform(both=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0].shape == (d["n_cal"], d["n_components"])
        assert result[1].shape == (d["n_cal"], d["n_components"])
        result_v = pls.transform(d["Xv"], d["yv"], both=True)
        assert isinstance(result_v, tuple) and len(result_v) == 2
        assert result_v[0].shape == (d["n_val"], d["n_components"])
        assert result_v[1].shape == (d["n_val"], d["n_components"])

    def test_fit_transform(self):
        pls = self.pls
        skl = self.skl
        d = self.d
        pls2 = PLSRegression(n_components=d["n_components"])
        x_scores = pls2.fit_transform(d["Xc"], d["yc"])
        assert isinstance(x_scores, scp.NDDataset)
        assert x_scores.shape == (d["n_cal"], d["n_components"])
        x_scores_skl = skl.fit(d["Xc"].data, d["yc"].data.squeeze()).transform(
            d["Xc"].data
        )
        assert_allclose(x_scores.data, x_scores_skl)

    def test_inverse_transform(self):
        pls = self.pls
        skl = self.skl
        d = self.d
        x_scores = pls.fit_transform(d["Xc"], d["yc"])
        x_hat = pls.inverse_transform(x_scores)
        assert isinstance(x_hat, scp.NDDataset)
        assert x_hat.shape == (d["n_cal"], d["n_features"])
        assert np.all(np.isfinite(x_hat.data))
        x_scores_skl = skl.fit(d["Xc"].data, d["yc"].data.squeeze()).transform(
            d["Xc"].data
        )
        x_hat_skl = skl.inverse_transform(x_scores_skl)
        assert_allclose(x_hat.data, x_hat_skl, rtol=1e-5)

    def test_inverse_transform_validation(self):
        pls = self.pls
        d = self.d
        x_scores_v = pls.transform(d["Xv"])
        xv_hat = pls.inverse_transform(x_scores_v)
        assert xv_hat.shape == (d["n_val"], d["n_features"])
        assert np.all(np.isfinite(xv_hat.data))


class TestPLSMultivariate:
    """PLS regression with multiple response variables."""

    @pytest.fixture(autouse=True)
    def _setup(self, pls_data):
        self.d = pls_data
        self.pls = PLSRegression(n_components=self.d["n_components"])
        self.pls.fit(self.d["Xc"], self.d["Yc"])
        self.skl = sklPLSRegression(n_components=self.d["n_components"])
        self.skl.fit(self.d["Xc"].data, self.d["Yc"].data)

    def test_fit_attributes(self):
        pls = self.pls
        d = self.d
        assert pls._fitted
        assert pls.x_loadings.shape == (d["n_components"], d["n_features"])
        assert pls.y_loadings.shape == (d["n_components"], d["n_targets"])
        assert pls.x_scores.shape == (d["n_cal"], d["n_components"])
        assert pls.y_scores.shape == (d["n_cal"], d["n_components"])
        assert np.all(np.isfinite(pls.x_loadings.data))
        assert np.all(np.isfinite(pls.y_loadings.data))

    def test_fit_against_sklearn(self):
        pls = self.pls
        skl = self.skl
        assert_allclose(pls.x_loadings.data, skl.x_loadings_.T)
        assert_allclose(pls.y_loadings.data, skl.y_loadings_.T)
        assert_allclose(pls.x_scores.data, skl.x_scores_)
        assert_allclose(pls.y_scores.data, skl.y_scores_)

    def test_predict(self):
        pls = self.pls
        d = self.d
        Y_hat = pls.predict(d["Xv"])
        assert isinstance(Y_hat, scp.NDDataset)
        assert Y_hat.shape == (d["n_val"], d["n_targets"])
        assert np.all(np.isfinite(Y_hat.data))

    def test_score(self):
        pls = self.pls
        skl = self.skl
        d = self.d
        assert_allclose(pls.score(), skl.score(d["Xc"].data, d["Yc"].data))
        score_val = pls.score(d["Xv"], d["Yv"])
        score_val_skl = skl.score(d["Xv"].data, d["Yv"].data)
        assert_allclose(score_val, score_val_skl, rtol=1e-5)
        assert 0.0 < score_val <= 1.0


class TestPLSMaskedData:
    """PLS with masked regions in X."""

    def test_masked_x_region(self, pls_data):
        d = pls_data
        n_comp = 2
        pls = PLSRegression(n_components=n_comp)
        Xc = d["Xc"].copy()
        mid_start = d["n_features"] // 3
        mid_end = 2 * d["n_features"] // 3
        Xc[:, mid_start:mid_end] = MASKED
        pls.fit(Xc, d["yc"])
        expected_cols = d["n_features"] - (mid_end - mid_start)
        assert pls._X.shape == (d["n_cal"], expected_cols)
        assert pls.X.shape == (d["n_cal"], d["n_features"])
