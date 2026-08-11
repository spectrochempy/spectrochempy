"""
Policy tests for deterministic analysis output metadata.

This suite validates the accepted maintainer policy
`spectrochempy_maintainer/rfcs/analysis-output-metadata-policy.md`
for representative decomposition outputs in the PR 2 runtime tranche:

- canonical derived identity (`name`, `title`, `description`, `filename`);
- canonical provenance/history text;
- independent copied metadata;
- fresh result timestamps distinct from source acquisition lineage;
- preserved structural support where the RFC says the source remains the
  scientific context authority.

The numerical outputs themselves are covered elsewhere; this file focuses on
the exact public metadata contract for representative families.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.decomposition.efa import EFA
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.analysis.decomposition.nmf import NMF
from spectrochempy.analysis.decomposition.pca import PCA
from spectrochempy.analysis.decomposition.svd import SVD
from spectrochempy.processing.transformation.npy import dot


@pytest.fixture
def semantic_dataset():
    """2D dataset with rich metadata for wrapper characterization."""
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
    ds.author = "test_author"
    ds.description = "source description"
    ds.origin = "test_origin"
    ds.filename = Path("source.spc")
    ds.acquisition_date = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    ds._created = datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    ds._modified = datetime(2021, 1, 2, 3, 4, 5, tzinfo=UTC)
    ds.meta.project = "analysis-output"
    ds.meta.tags = ["baseline"]
    ds.history = ["original history entry"]
    return ds


@pytest.fixture
def efa_dataset():
    """Smooth mixture dataset suitable for EFA and NMF characterization."""
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
        coordset=[
            scp.Coord(time, title="time", units="min"),
            scp.Coord(wavelength, title="wavelength", units="nm"),
        ],
        units="absorbance",
        title="mixture",
        name="mixture_name",
    )
    ds.history = ["mixture history"]
    return ds


@pytest.fixture
def mcrals_inputs():
    """Small exact-factorization problem for MCRALS output characterization."""
    time = scp.Coord(np.arange(6), title="elution time", units="hours")
    wavelength = scp.Coord(np.arange(8), title="wavelength", units="cm^-1")
    species = scp.Coord(range(2), title="species", labels=["PS#0", "PS#1"])
    c0 = scp.NDDataset(
        np.column_stack([np.linspace(1.0, 2.0, 6), np.linspace(2.0, 1.0, 6)]),
        coordset=(time, species),
    )
    st0 = scp.NDDataset(
        np.vstack([np.linspace(1.0, 2.0, 8), np.linspace(2.0, 1.0, 8)]),
        coordset=(species, wavelength),
    )
    data = dot(c0, st0)
    data.title = "intensity"
    data.units = "absorbance"
    data.name = "mcr_source"
    return data, c0


class TestLatentOutputs:
    """Characterize latent analysis outputs."""

    def test_pca_transform_synthesizes_component_axis(self, semantic_dataset):
        scores = PCA(n_components=2).fit(semantic_dataset).transform()

        assert scores.shape == (semantic_dataset.shape[0], 2)
        assert scores.dims == ["y", "k"]
        assert scores.y.title == semantic_dataset.y.title
        assert scores.y.units == semantic_dataset.y.units
        assert scores.k.title == "components"
        assert list(scores.k.labels) == ["PC1", "PC2"]

    def test_pca_components_preserve_feature_axis_and_synthesize_k(
        self, semantic_dataset
    ):
        loadings = PCA(n_components=2).fit(semantic_dataset).components

        assert loadings.shape == (2, semantic_dataset.shape[1])
        assert loadings.dims == ["k", "x"]
        assert loadings.k.title == "components"
        assert loadings.x.title == semantic_dataset.x.title
        assert loadings.x.units == semantic_dataset.x.units

    def test_nmf_transform_matches_latent_output_contract(self, efa_dataset):
        scores = (
            NMF(
                n_components=2,
                init="nndsvda",
                max_iter=500,
                random_state=0,
            )
            .fit(efa_dataset)
            .transform()
        )

        assert scores.shape == (efa_dataset.shape[0], 2)
        assert scores.dims == ["y", "k"]
        assert scores.y.title == efa_dataset.y.title
        assert scores.y.units == efa_dataset.y.units
        assert scores.k.title == "components"

    def test_efa_forward_eigenvalues_use_observation_and_component_axes(
        self, efa_dataset
    ):
        f_ev = EFA(n_components=2).fit(efa_dataset).f_ev

        assert f_ev.shape == (efa_dataset.shape[0], efa_dataset.shape[1])
        assert f_ev.dims == ["y", "k"]
        assert f_ev.y.title == efa_dataset.y.title
        assert f_ev.y.units == efa_dataset.y.units
        assert f_ev.k.title == "components"

    def test_mcrals_profiles_follow_component_axis_convention(self, mcrals_inputs):
        dataset, c0 = mcrals_inputs
        mcr = MCRALS(tol=50.0).fit(dataset, c0)

        assert mcr.C.shape == (dataset.shape[0], 2)
        assert mcr.C.dims == ["y", "k"]
        assert mcr.C.y.title == dataset.y.title
        assert mcr.C.k.title == "components"

        assert mcr.St.shape == (2, dataset.shape[1])
        assert mcr.St.dims == ["k", "x"]
        assert mcr.St.k.title == "components"
        assert mcr.St.x.title == dataset.x.title


class TestMetadataAndProvenance:
    """Validate canonical metadata propagation on representative outputs."""

    def test_transform_applies_canonical_scores_metadata(self, semantic_dataset):
        before = datetime.now(UTC)
        scores = PCA(n_components=2).fit(semantic_dataset).transform()
        after = datetime.now(UTC)

        assert scores.name == "source_name_PCA.scores"
        assert scores.title == "scores"
        assert scores.description == "scores from PCA fit of source_name."
        assert scores.units is None
        assert scores.author == semantic_dataset.author
        assert scores.origin == semantic_dataset.origin
        assert scores.filename is None
        assert scores.meta.project == semantic_dataset.meta.project
        assert scores.meta.tags == semantic_dataset.meta.tags
        assert scores.meta is not semantic_dataset.meta
        assert scores.acquisition_date == semantic_dataset.acquisition_date
        assert (
            before - timedelta(seconds=1)
            <= scores._created
            <= after + timedelta(seconds=1)
        )
        assert (
            before - timedelta(seconds=1)
            <= scores._modified
            <= after + timedelta(seconds=1)
        )
        assert scores.history[-1].endswith(
            "Created analysis output scores with PCA from source_name."
        )

    def test_components_apply_canonical_component_identity(self, semantic_dataset):
        loadings = PCA(n_components=2).fit(semantic_dataset).components

        assert loadings.name == "source_name_PCA.components"
        assert loadings.title == "components"
        assert loadings.units is None
        assert loadings.description == "components from PCA fit of source_name."
        assert loadings.filename is None
        assert loadings.history[-1].endswith(
            "Created analysis output components with PCA from source_name."
        )

    def test_diagnostic_output_uses_canonical_diagnostic_text_and_metadata_copy(
        self, semantic_dataset
    ):
        ev_ratio = PCA(n_components=2).fit(semantic_dataset).ev_ratio

        assert ev_ratio.title == "explained variance ratio"
        assert ev_ratio.name == "source_name_PCA.explained_variance_ratio"
        assert ev_ratio.description == (
            "explained variance ratio diagnostic from PCA fit of source_name."
        )
        assert ev_ratio.units == "percent"
        assert ev_ratio.author == semantic_dataset.author
        assert ev_ratio.meta.project == semantic_dataset.meta.project
        assert ev_ratio.meta is not semantic_dataset.meta
        assert ev_ratio.filename is None
        assert ev_ratio.history[-1].endswith(
            "Created diagnostic explained_variance_ratio with PCA from source_name."
        )


class TestReconstructionOutputs:
    """Validate source-geometry reconstructions as derived outputs."""

    def test_pca_inverse_transform_returns_source_geometry_and_context(
        self, semantic_dataset
    ):
        reconstructed = PCA(n_components=2).fit(semantic_dataset).inverse_transform()

        assert reconstructed.shape == semantic_dataset.shape
        assert reconstructed.dims == semantic_dataset.dims
        assert reconstructed.y.title == semantic_dataset.y.title
        assert reconstructed.x.title == semantic_dataset.x.title
        assert reconstructed.title == "reconstruction"
        assert reconstructed.units == semantic_dataset.units
        assert reconstructed.name == "source_name_PCA.reconstruction"
        assert reconstructed.description == (
            "reconstruction from PCA fit of source_name."
        )
        assert reconstructed.filename is None
        assert reconstructed.history[-1].endswith(
            "Created analysis output reconstruction with PCA from source_name."
        )

    def test_nmf_inverse_transform_returns_source_geometry_and_context(
        self, efa_dataset
    ):
        reconstructed = (
            NMF(
                n_components=2,
                init="nndsvda",
                max_iter=500,
                random_state=0,
            )
            .fit(efa_dataset)
            .inverse_transform()
        )

        assert reconstructed.shape == efa_dataset.shape
        assert reconstructed.dims == efa_dataset.dims
        assert reconstructed.y.title == efa_dataset.y.title
        assert reconstructed.x.title == efa_dataset.x.title
        assert reconstructed.title == "reconstruction"
        assert reconstructed.units == efa_dataset.units


class TestDiagnosticOutputs:
    """Validate diagnostic analysis datasets."""

    def test_pca_explained_variance_ratio_is_component_vector(self, semantic_dataset):
        ev_ratio = PCA(n_components=2).fit(semantic_dataset).ev_ratio

        assert ev_ratio.shape == (2,)
        assert ev_ratio.dims == ["k"]
        assert ev_ratio.k.title == "components"
        assert ev_ratio.units == "percent"

    def test_svd_singular_values_are_component_vector_diagnostics(
        self, semantic_dataset
    ):
        sv = SVD().fit(semantic_dataset).sv

        assert sv.shape == (semantic_dataset.shape[1],)
        assert sv.dims == ["k"]
        assert sv.k.title == "components"
        assert sv.title == "singular values"
        assert sv.name == "source_name_SVD.singular_values"
        assert sv.description == (
            "singular values diagnostic from SVD fit of source_name."
        )
        assert sv.author == semantic_dataset.author
        assert sv.filename is None
        assert sv.units is None
        assert sv.history[-1].endswith(
            "Created diagnostic singular_values with SVD from source_name."
        )


class TestCurrentApiLimits:
    """Characterize current analysis API limitations where relevant."""

    def test_svd_transform_is_not_implemented(self, semantic_dataset):
        svd = SVD().fit(semantic_dataset)

        with pytest.raises(NotImplementedError):
            svd.transform()
