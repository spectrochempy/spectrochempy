# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Permanent tests for the analysis source-provenance snapshot.

These tests validate the G14 provenance behavior of the accepted maintainer
policy `spectrochempy_maintainer/rfcs/analysis-output-metadata-policy.md`
(PR 1: provenance snapshot and path consistency):

- stored single-source analysis outputs preserve the exact ``author`` of the
  scientific source instead of recreating the runtime user/host value through
  ``NDDatasetType`` / ``NDDataset(value)`` coercion;
- the provenance snapshot is captured once at ``fit`` time; a direct
  ``NDDataset`` argument is authoritative for that call only and never
  overwrites the fitted-model snapshot (model vs call provenance are kept
  distinct);
- ``fit`` always replaces or clears the snapshots from its new inputs, so no
  obsolete metadata survives a refit;
- X-side and Y-side outputs use their own scientific source;
- supervised ``predict`` now follows the accepted multi-source PR 2 policy
  (`Xtrain`, `Ytrain`, `Xpredict`) without leaking stale fitted provenance;
- input datasets are never mutated and output metadata is independent of later
  source mutation.

The PR 1 author guarantees remain covered here and are complemented by the PR 2
deterministic identity/provenance/history rules where the accepted policy now
requires them.
"""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import AnalysisConfigurable
from spectrochempy.analysis.crossdecomposition.pls import PLSRegression
from spectrochempy.analysis.decomposition.efa import EFA
from spectrochempy.analysis.decomposition.fast_ica import FastICA
from spectrochempy.analysis.decomposition.nmf import NMF
from spectrochempy.analysis.decomposition.pca import PCA
from spectrochempy.utils.system import get_user_and_node


@pytest.fixture()
def source_dataset():
    """Small PCA dataset with rich metadata and an explicit author."""
    y = scp.Coord(np.linspace(0.0, 5.0, 6), title="time", units="s")
    x = scp.Coord(np.linspace(400.0, 700.0, 4), title="wavelength", units="nm")
    ds = scp.NDDataset(
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ),
        coordset=[y, x],
        units="absorbance",
        title="source title",
        name="source_name",
    )
    ds.author = "source_author"
    ds.meta.project = "provenance"
    ds.history = ["original history entry"]
    return ds


@pytest.fixture()
def other_author_dataset(source_dataset):
    """A different scientific source with a distinct author."""
    other = source_dataset.copy()
    other.author = "other_author"
    other.name = "other_name"
    other.title = "other title"
    return other


@pytest.fixture()
def pls_inputs():
    """PLSRegression X/Y training inputs with distinct authors."""
    rng = np.random.default_rng(0)
    X = scp.NDDataset(rng.normal(size=(20, 5)), title="X training")
    X.author = "x_author"
    Y = scp.NDDataset(rng.normal(size=(20, 3)), title="Y training")
    Y.author = "y_author"
    return X, Y


@pytest.fixture()
def mixture_dataset():
    """Smooth mixture dataset suitable for NMF/FastICA/EFA."""
    time = np.linspace(0.0, 1.0, 12)
    wavelength = np.linspace(400.0, 700.0, 6)
    concentrations = np.column_stack(
        [
            np.exp(-0.5 * ((time - 0.30) / 0.10) ** 2),
            0.7 * np.exp(-0.5 * ((time - 0.70) / 0.12) ** 2),
        ]
    )
    spectra = np.vstack(
        [
            1.0 + 0.2 * np.cos(np.linspace(0.0, np.pi, wavelength.size)),
            0.8 + 0.3 * np.sin(np.linspace(0.0, np.pi, wavelength.size)),
        ]
    )
    ds = scp.NDDataset(
        concentrations @ spectra,
        coordset=[scp.Coord(time), scp.Coord(wavelength)],
        title="mixture",
    )
    ds.author = "mixture_author"
    return ds


class TestSourceNonMutation:
    """Prove that analysis does not mutate its scientific sources."""

    def test_fit_and_outputs_do_not_mutate_source(self, source_dataset):
        ds = source_dataset
        before = (
            ds.data.copy(),
            ds.mask.copy(),
            ds.author,
            list(ds.history),
            ds.meta.project,
            ds.title,
            ds.name,
            list(ds.dims),
        )

        pca = PCA(n_components=2).fit(ds)
        _ = pca.transform()
        _ = pca.scores
        _ = pca.result.scores
        _ = pca.components
        _ = pca.ev_ratio
        _ = pca.inverse_transform()

        assert np.array_equal(ds.data, before[0])
        assert np.array_equal(ds.mask, before[1])
        assert ds.author == before[2]
        assert ds.history == before[3]
        assert ds.meta.project == before[4]
        assert ds.title == before[5]
        assert ds.name == before[6]
        assert ds.dims == before[7]

    def test_output_metadata_independent_of_later_source_mutation(self, source_dataset):
        ds = source_dataset
        pca = PCA(n_components=2).fit(ds)
        scores = pca.scores
        components = pca.components

        # Mutating the source after the snapshot must not affect the outputs.
        ds.author = "mutated_after_fit"
        ds.meta["injected"] = "value"
        ds.history = ["mutated history"]
        ds.title = "mutated title"

        assert scores.author == "source_author"
        assert components.author == "source_author"
        assert scores.meta is not ds.meta
        assert scores.meta.project == "provenance"
        assert "injected" not in scores.meta
        assert scores.history[-1].endswith(
            "Created analysis output scores with PCA from source_name."
        )


class TestPCAMonoSource:
    """PCA stored and direct paths agree on the scientific source author."""

    def test_stored_outputs_preserve_source_author(self, source_dataset):
        pca = PCA(n_components=2).fit(source_dataset)

        assert pca.transform().author == "source_author"
        assert pca.scores.author == "source_author"
        assert pca.result.scores.author == "source_author"
        assert pca.components.author == "source_author"
        assert pca.loadings.author == "source_author"
        assert pca.ev_ratio.author == "source_author"
        assert pca.ev.author == "source_author"
        assert pca.inverse_transform().author == "source_author"

    def test_direct_call_uses_argument_author(
        self, source_dataset, other_author_dataset
    ):
        pca = PCA(n_components=2).fit(source_dataset)

        assert pca.transform(other_author_dataset).author == "other_author"
        assert pca.inverse_transform(pca.transform(other_author_dataset)).author == (
            "other_author"
        )

    def test_fit_transform_uses_direct_author(self, other_author_dataset):
        scores = PCA(n_components=2).fit_transform(other_author_dataset)
        assert scores.author == "other_author"

    def test_refit_replaces_snapshot(self, source_dataset, other_author_dataset):
        pca = PCA(n_components=2)
        pca.fit(source_dataset)
        assert pca.scores.author == "source_author"

        pca.fit(other_author_dataset)
        assert pca.scores.author == "other_author"

    def test_fit_snapshot_survives_direct_transform(
        self, source_dataset, other_author_dataset
    ):
        pca = PCA(n_components=2).fit(source_dataset)

        # The direct call is authoritative for its own output only.
        assert pca.transform(other_author_dataset).author == "other_author"

        # The fit snapshot is untouched: components and stored paths keep
        # the provenance of the fitted model, not of the transformed data.
        assert pca.components.author == "source_author"
        assert pca.scores.author == "source_author"
        assert pca.transform().author == "source_author"

    def test_array_like_input_has_no_stale_source_author(self, source_dataset):
        pca = PCA(n_components=2).fit(source_dataset.data)

        assert pca._X_source_metadata is None
        assert pca.scores.author == get_user_and_node()
        assert pca.transform(source_dataset.data).author == get_user_and_node()

    def test_direct_array_like_does_not_leak_snapshot_author(self, source_dataset):
        pca = PCA(n_components=2).fit(source_dataset)

        # A direct array-like argument carries no scientific provenance: even
        # though a fit snapshot exists, it must not leak into the output.
        assert pca.transform(source_dataset.data).author == get_user_and_node()

        # The snapshot is untouched and still serves the stored paths.
        assert pca.scores.author == "source_author"
        assert pca.transform().author == "source_author"

    def test_refit_without_y_clears_y_snapshot(self, source_dataset):
        Y = source_dataset.copy()
        Y.author = "stale_y_author"

        pca = PCA(n_components=2)
        # Exercise the base fit signature that accepts Y (not exposed by PCA).
        AnalysisConfigurable.fit(pca, source_dataset, Y)
        assert pca._X_source_metadata is not None
        assert pca._X_source_metadata.author == "source_author"
        assert pca._Y_source_metadata is not None
        assert pca._Y_source_metadata.author == "stale_y_author"

        # A new fit without Y must not keep the obsolete Y snapshot.
        pca.fit(source_dataset)
        assert pca._X_source_metadata is not None
        assert pca._X_source_metadata.author == "source_author"
        assert pca._Y_source_metadata is None

    def test_refit_from_array_like_clears_snapshots(self, source_dataset):
        pca = PCA(n_components=2).fit(source_dataset)
        assert pca._X_source_metadata is not None

        # Array-like inputs carry no scientific metadata: the snapshots must
        # be cleared, never inherited from a previous dataset fit.
        pca.fit(source_dataset.data)
        assert pca._X_source_metadata is None
        assert pca._Y_source_metadata is None


class TestPLSRegressionYside:
    """PLS X-side and Y-side outputs use their own scientific source."""

    def test_x_and_y_snapshots_are_distinct(self, pls_inputs):
        X, Y = pls_inputs
        pls = PLSRegression(n_components=2).fit(X, Y)

        assert pls._X_source_metadata is not None
        assert pls._Y_source_metadata is not None
        assert pls._X_source_metadata.author == "x_author"
        assert pls._Y_source_metadata.author == "y_author"
        assert pls._X_source_metadata is not pls._Y_source_metadata

    def test_x_side_outputs_use_x_author(self, pls_inputs):
        X, Y = pls_inputs
        pls = PLSRegression(n_components=2).fit(X, Y)

        for output in (
            pls.x_scores,
            pls.x_loadings,
            pls.x_weights,
            pls.x_rotations,
            pls.result.x_scores,
            pls.result.x_loadings,
        ):
            assert output.author == "x_author"

    def test_y_side_outputs_use_y_author(self, pls_inputs):
        X, Y = pls_inputs
        pls = PLSRegression(n_components=2).fit(X, Y)

        for output in (
            pls.y_scores,
            pls.y_loadings,
            pls.y_weights,
            pls.y_rotations,
            pls.result.y_scores,
            pls.result.y_loadings,
        ):
            assert output.author == "y_author"

    def test_predict_merges_training_and_prediction_provenance(self, pls_inputs):
        X, Y = pls_inputs
        X.name = "xtrain"
        X.origin = "origin_xtrain"
        Y.name = "ytrain"
        Y.origin = "origin_ytrain"
        pls = PLSRegression(n_components=2).fit(X, Y)

        prediction = pls.predict()
        assert prediction.author == "x_author & y_author"
        assert prediction.origin == "origin_xtrain & origin_ytrain"
        assert prediction.title == "prediction"
        assert prediction.name == "xtrain_PLSRegression.prediction"
        assert prediction.description == (
            "Prediction from PLSRegression fit of xtrain + ytrain applied to xtrain."
        )
        assert prediction.history[-1].endswith(
            "Created analysis output prediction with PLSRegression from "
            "xtrain + ytrain; applied to xtrain."
        )
        assert prediction.filename is None
        assert prediction.meta is not Y.meta

        Xnew = X.copy()
        Xnew.author = "predict_author"
        Xnew.origin = "origin_predict"
        Xnew.name = "xpredict"
        direct_prediction = pls.predict(Xnew)
        assert direct_prediction.author == "x_author & y_author & predict_author"
        assert (
            direct_prediction.origin == "origin_xtrain & origin_ytrain & origin_predict"
        )
        assert direct_prediction.name == "xpredict_PLSRegression.prediction"
        assert direct_prediction.description == (
            "Prediction from PLSRegression fit of xtrain + ytrain applied to xpredict."
        )
        assert direct_prediction.history[-1].endswith(
            "Created analysis output prediction with PLSRegression from "
            "xtrain + ytrain; applied to xpredict."
        )

    def test_transform_both_uses_respective_authors(self, pls_inputs):
        X, Y = pls_inputs
        pls = PLSRegression(n_components=2).fit(X, Y)

        x_scores, y_scores = pls.transform(X, Y, both=True)
        assert x_scores.author == "x_author"
        assert y_scores.author == "y_author"

    def test_predict_direct_array_like_does_not_leak_old_dataset_provenance(
        self, pls_inputs
    ):
        X, Y = pls_inputs
        X.name = "xtrain"
        Y.name = "ytrain"
        pls = PLSRegression(n_components=2).fit(X, Y)

        prediction = pls.predict(X.data)
        assert prediction.author == "x_author & y_author"
        assert prediction.name == "PLSRegression.prediction"
        assert prediction.description == (
            "Prediction from PLSRegression fit of xtrain + ytrain applied to <unnamed>."
        )
        assert prediction.history[-1].endswith(
            "Created analysis output prediction with PLSRegression from "
            "xtrain + ytrain; applied to <unnamed>."
        )

    def test_transform_direct_y_argument_is_authoritative(self, pls_inputs):
        X, Y = pls_inputs
        pls = PLSRegression(n_components=2).fit(X, Y)

        # A direct Y argument is authoritative for the Y-side output of that
        # call only, even when its author differs from the fitted Y source.
        Xother = X.copy()
        Xother.author = "x_other_author"
        Yother = Y.copy()
        Yother.author = "y_other_author"

        x_scores, y_scores = pls.transform(Xother, Yother, both=True)
        assert x_scores.author == "x_other_author"
        assert y_scores.author == "y_other_author"

        # The fit snapshots are untouched by the direct call.
        assert pls._X_source_metadata.author == "x_author"
        assert pls._Y_source_metadata.author == "y_author"

        # Without a direct Y argument, the Y-side output falls back to the
        # fitted Y snapshot, while the direct X argument stays authoritative
        # for the X-side output.
        x_scores, y_scores = pls.transform(Xother, both=True)
        assert x_scores.author == "x_other_author"
        assert y_scores.author == "y_author"

    def test_direct_array_like_y_does_not_leak_snapshot_author(self, pls_inputs):
        X, Y = pls_inputs
        pls = PLSRegression(n_components=2).fit(X, Y)

        # Direct array-like X and Y carry no scientific provenance: the fitted
        # snapshots must not leak into either side of the output.
        x_scores, y_scores = pls.transform(X.data, Y.data, both=True)
        assert x_scores.author == get_user_and_node()
        assert y_scores.author == get_user_and_node()

    def test_inverse_transform_direct_y_transform_authority(self, pls_inputs):
        X, Y = pls_inputs
        pls = PLSRegression(n_components=2).fit(X, Y)

        Xt = X.copy()
        Xt.author = "x_transform_author"
        Yt = Y.copy()
        Yt.author = "y_transform_author"
        x_scores, y_scores = pls.transform(Xt, Yt, both=True)

        # The Y_transform keyword is recognized as the direct Y argument and
        # is authoritative for the Y-side reconstruction of this call.
        X_rec, Y_rec = pls.inverse_transform(x_scores, y_scores, both=True)
        assert X_rec.author == "x_transform_author"
        assert Y_rec.author == "y_transform_author"

    def test_fit_transform_uses_direct_authors(self, pls_inputs):
        X, Y = pls_inputs
        pls = PLSRegression(n_components=2)
        x_scores = pls.fit_transform(X, Y)
        assert x_scores.author == "x_author"


class TestOtherDecompositionFamilies:
    """NMF, FastICA and EFA preserve the source author on stored outputs."""

    def test_nmf(self, mixture_dataset):
        nmf = NMF(
            n_components=2,
            init="nndsvda",
            max_iter=500,
            random_state=0,
        ).fit(mixture_dataset)

        assert nmf.transform().author == "mixture_author"
        assert nmf.components.author == "mixture_author"
        assert nmf.result.W.author == "mixture_author"

    def test_fast_ica(self, mixture_dataset):
        ica = FastICA(
            n_components=2,
            max_iter=500,
            random_state=0,
        ).fit(mixture_dataset)

        assert ica.A.author == "mixture_author"
        assert ica.components.author == "mixture_author"
        assert ica.result.A.author == "mixture_author"

    def test_efa(self, mixture_dataset):
        efa = EFA(n_components=2).fit(mixture_dataset)

        assert efa.f_ev.author == "mixture_author"
        assert efa.result.f_ev.author == "mixture_author"
