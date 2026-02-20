# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for plot_score composite function."""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestPlotScore:
    """Tests for plot_score composite function."""

    def test_plot_score_2d_basic(self):
        """Test basic 2D score plot functionality."""
        from spectrochempy.plotting.composite import plot_score

        scores = np.random.randn(50, 5)

        ax = plot_score(scores, components=(1, 2), show=False)

        assert ax is not None
        assert ax.get_xlabel() == "PC1"
        assert ax.get_ylabel() == "PC2"

        assert len(ax.collections) == 1
        assert len(ax.collections[0].get_offsets()) == 50

        plt.close()

    def test_plot_score_3d_basic(self):
        """Test basic 3D score plot functionality."""
        from spectrochempy.plotting.composite import plot_score

        scores = np.random.randn(50, 5)

        ax = plot_score(scores, components=(1, 2, 3), show=False)

        assert ax is not None
        assert hasattr(ax, "get_zlabel")

        plt.close()

    def test_pca_plot_score_wrapper(self):
        """Test PCA.plot_score() wrapper."""
        import spectrochempy as scp

        X = scp.read("irdata/nh4y-activation.spg")
        pca = scp.PCA(n_components=5)
        pca.fit(X)

        ax = pca.plot_score((1, 2), show=False)

        assert ax is not None

        plt.close()

    def test_pca_scoreplot_deprecated(self):
        """Test PCA.scoreplot() emits DeprecationWarning."""
        import spectrochempy as scp

        X = scp.read("irdata/nh4y-activation.spg")
        pca = scp.PCA(n_components=5)
        pca.fit(X)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = pca.scoreplot(1, 2, show=False)

            deprecation_warnings = [
                item for item in w if issubclass(item.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

        assert ax is not None

        plt.close()

    def test_invalid_components_length(self):
        """Test that invalid components length raises ValueError."""
        from spectrochempy.plotting.composite import plot_score

        scores = np.random.randn(50, 5)

        with pytest.raises(ValueError, match="length 2 or 3"):
            plot_score(scores, components=(1,), show=False)

        with pytest.raises(ValueError, match="length 2 or 3"):
            plot_score(scores, components=(1, 2, 3, 4), show=False)

    def test_component_out_of_range(self):
        """Test that out-of-range components raise ValueError."""
        from spectrochempy.plotting.composite import plot_score

        scores = np.random.randn(50, 5)

        with pytest.raises(ValueError, match="out of range"):
            plot_score(scores, components=(1, 10), show=False)

        with pytest.raises(ValueError, match="out of range"):
            plot_score(scores, components=(0, 2), show=False)

    def test_plot_score_with_custom_color(self):
        """Test plot_score with fixed color."""
        from spectrochempy.plotting.composite import plot_score

        scores = np.random.randn(50, 5)

        ax = plot_score(scores, components=(1, 2), color="red", show=False)

        assert ax is not None

        plt.close()

    def test_plot_score_with_ax(self):
        """Test plot_score with provided axes."""
        from spectrochempy.plotting.composite import plot_score

        scores = np.random.randn(50, 5)

        fig, ax = plt.subplots()
        result = plot_score(scores, components=(1, 2), ax=ax, show=False)

        assert result is ax

        plt.close()

    def test_plot_score_labels_2d(self):
        """Test plot_score with labels in 2D."""
        import spectrochempy as scp
        from spectrochempy.plotting.composite import plot_score

        scores_data = np.random.randn(10, 5)
        scores = scp.NDDataset(scores_data)

        labels = np.array([f"S{i}" for i in range(10)])
        scores.y = scp.Coord(labels=labels.reshape(-1, 1))

        ax = plot_score(scores, components=(1, 2), show_labels=True, show=False)

        text_objects = [
            child for child in ax.get_children() if hasattr(child, "get_text")
        ]
        text_objects = [t for t in text_objects if t.get_text()]
        assert len(text_objects) == 10

        plt.close()

    def test_plot_score_labels_3d(self):
        """Test plot_score with labels in 3D."""
        import spectrochempy as scp
        from spectrochempy.plotting.composite import plot_score

        scores_data = np.random.randn(10, 5)
        scores = scp.NDDataset(scores_data)

        labels = np.array([f"S{i}" for i in range(10)])
        scores.y = scp.Coord(labels=labels.reshape(-1, 1))

        ax = plot_score(scores, components=(1, 2, 3), show_labels=True, show=False)

        assert ax is not None

        plt.close()

    def test_plot_score_labels_column(self):
        """Test plot_score with labels_column selection."""
        import spectrochempy as scp
        from spectrochempy.plotting.composite import plot_score

        scores_data = np.random.randn(10, 5)
        scores = scp.NDDataset(scores_data)

        labels_col0 = np.array([f"A{i}" for i in range(10)])
        labels_col1 = np.array([f"B{i}" for i in range(10)])
        labels = np.column_stack([labels_col0, labels_col1])
        scores.y = scp.Coord(labels=labels)

        ax = plot_score(
            scores, components=(1, 2), show_labels=True, labels_column=1, show=False
        )

        assert ax is not None

        plt.close()

    def test_plot_score_invalid_labels_column(self):
        """Test that invalid labels_column raises ValueError."""
        import spectrochempy as scp
        from spectrochempy.plotting.composite import plot_score

        scores_data = np.random.randn(10, 5)
        scores = scp.NDDataset(scores_data)

        labels = np.array([f"S{i}" for i in range(10)])
        scores.y = scp.Coord(labels=labels.reshape(-1, 1))

        with pytest.raises(ValueError, match="labels_column"):
            plot_score(
                scores,
                components=(1, 2),
                show_labels=True,
                labels_column=5,
                show=False,
            )

        plt.close()

    def test_plot_score_no_labels(self):
        """Test that show_labels=True without labels raises ValueError."""
        import spectrochempy as scp
        from spectrochempy.plotting.composite import plot_score

        scores_data = np.random.randn(10, 5)
        scores = scp.NDDataset(scores_data)

        with pytest.raises(ValueError, match="no y coordinate"):
            plot_score(scores, components=(1, 2), show_labels=True, show=False)

        plt.close()

    def test_plot_score_no_labels_attribute(self):
        """Test that show_labels=True with empty labels raises ValueError."""
        from spectrochempy.plotting.composite import plot_score

        scores = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="no y coordinate"):
            plot_score(scores, components=(1, 2), show_labels=True, show=False)

        plt.close()

    def test_plot_score_with_external_scores_labels(self):
        """Test plot_score with external scores object that has custom labels."""
        import spectrochempy as scp

        X = scp.read("irdata/nh4y-activation.spg")
        pca = scp.PCA(n_components=5)
        pca.fit(X)

        scores = pca.transform()

        custom_labels = np.array([f"Sample_{i}" for i in range(scores.shape[0])])
        scores.y.labels = custom_labels.reshape(-1, 1)

        ax = pca.plot_score(
            scores=scores,
            components=(1, 2),
            show_labels=True,
            labels_column=0,
            show=False,
        )

        text_objects = [
            child
            for child in ax.get_children()
            if hasattr(child, "get_text") and child.get_text()
        ]
        assert len(text_objects) == scores.shape[0]

        plt.close()

    def test_plot_score_external_scores_appended_labels(self):
        """Test plot_score with scores that have appended label columns."""
        import spectrochempy as scp

        X = scp.read("irdata/nh4y-activation.spg")
        pca = scp.PCA(n_components=5)
        pca.fit(X)

        scores = pca.transform()

        n_samples = scores.shape[0]
        custom_labels = np.array([f"C{i}" for i in range(n_samples)]).reshape(-1, 1)
        scores.y.labels = custom_labels

        ax = pca.plot_score(
            scores=scores,
            show_labels=True,
            labels_column=2,
            show=False,
        )

        text_objects = [
            child
            for child in ax.get_children()
            if hasattr(child, "get_text") and child.get_text()
        ]
        assert len(text_objects) == n_samples

        plt.close()

    def test_plot_score_default_scores_still_works(self):
        """Test that plot_score still works without explicit scores argument."""
        import spectrochempy as scp

        X = scp.read("irdata/nh4y-activation.spg")
        pca = scp.PCA(n_components=5)
        pca.fit(X)

        ax = pca.plot_score((1, 2), show=False)

        assert ax is not None
        assert ax.get_xlabel() == "PC1"
        assert ax.get_ylabel() == "PC2"

        plt.close()
