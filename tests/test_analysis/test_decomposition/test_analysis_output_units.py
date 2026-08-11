# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""G15 units policy tests for analysis-output datasets."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.crossdecomposition.pls import PLSRegression
from spectrochempy.analysis.decomposition.efa import EFA
from spectrochempy.analysis.decomposition.fast_ica import FastICA
from spectrochempy.analysis.decomposition.nmf import NMF
from spectrochempy.analysis.decomposition.pca import PCA
from spectrochempy.analysis.decomposition.simplisma import SIMPLISMA
from spectrochempy.analysis.decomposition.svd import SVD


@pytest.fixture()
def pls_unit_inputs():
    rng = np.random.default_rng(0)
    X = scp.NDDataset(rng.normal(size=(20, 5)), title="X training", units="absorbance")
    Y = scp.NDDataset(rng.normal(size=(20, 3)), title="Y training", units="volt")
    return X, Y


def test_pca_latent_outputs_are_unitless(low_rank_pca_dataset):
    pca = PCA(n_components=2).fit(low_rank_pca_dataset)

    assert pca.scores.units is None
    assert pca.components.units is None
    assert pca.loadings.units is None
    assert pca.explained_variance.units is None
    assert pca.explained_variance_ratio.units == "percent"
    assert pca.cumulative_explained_variance.units == "percent"


def test_nmf_latent_outputs_and_reconstruction_follow_g15(efa_dataset):
    nmf = NMF(
        n_components=2,
        init="nndsvda",
        max_iter=500,
        random_state=0,
    ).fit(efa_dataset)

    assert nmf.transform().units is None
    assert nmf.components.units is None
    assert nmf.inverse_transform().units == efa_dataset.units


def test_fastica_and_efa_outputs_are_unitless(fastica_dataset, efa_dataset):
    ica = FastICA(n_components=4, random_state=0, whiten="unit-variance").fit(
        fastica_dataset
    )
    efa = EFA(n_components=2).fit(efa_dataset)

    assert ica.A.units is None
    assert ica.St.units is None
    assert ica.components.units is None
    assert efa.transform().units is None
    assert efa.components.units is None


def test_simplisma_profiles_are_unitless(simplisma_dataset):
    sma = SIMPLISMA(n_components=2).fit(simplisma_dataset)

    assert sma.C.units is None
    assert sma.St.units is None


def test_pls_latent_outputs_are_unitless(pls_unit_inputs):
    X, Y = pls_unit_inputs
    pls = PLSRegression(n_components=2).fit(X, Y)

    for output in (
        pls.x_scores,
        pls.y_scores,
        pls.x_loadings,
        pls.y_loadings,
        pls.x_weights,
        pls.y_weights,
        pls.x_rotations,
        pls.y_rotations,
    ):
        assert output.units is None


def test_pls_predictions_preserve_y_units(pls_unit_inputs):
    X, Y = pls_unit_inputs
    pls = PLSRegression(n_components=2).fit(X, Y)

    assert pls.predict().units == Y.units

    Xnew = X.copy()
    Xnew.units = "counts"
    assert pls.predict(Xnew).units == Y.units
    assert pls.predict(X.data).units == Y.units


def test_pls_predictions_preserve_none_y_units(pls_unit_inputs):
    X, Y = pls_unit_inputs
    Y = Y.to(None, force=True)
    pls = PLSRegression(n_components=2).fit(X, Y)

    assert pls.predict().units is None
    assert pls.predict(X.copy()).units is None
    assert pls.predict(X.data).units is None


def test_pls_refit_replaces_prediction_units(pls_unit_inputs):
    X, Y = pls_unit_inputs
    pls = PLSRegression(n_components=2).fit(X, Y)
    assert pls.predict().units == Y.units

    Y2 = scp.NDDataset(Y.data.copy(), title="Y training", units="ampere")
    pls.fit(X, Y2)
    assert pls.predict().units == Y2.units


def test_reconstruction_units_preserve_none(low_rank_pca_dataset, efa_dataset):
    pca_input = low_rank_pca_dataset.to(None, force=True)
    nmf_input = efa_dataset.to(None, force=True)

    assert PCA(n_components=2).fit(pca_input).inverse_transform().units is None
    assert (
        NMF(
            n_components=2,
            init="nndsvda",
            max_iter=500,
            random_state=0,
        )
        .fit(nmf_input)
        .inverse_transform()
        .units
        is None
    )


def test_svd_diagnostics_follow_g15(low_rank_pca_dataset):
    svd = SVD().fit(low_rank_pca_dataset)

    assert svd.singular_values.units is None
    assert svd.explained_variance.units is None
    assert svd.explained_variance_ratio.units == "percent"
    assert svd.cumulative_explained_variance.units == "percent"
