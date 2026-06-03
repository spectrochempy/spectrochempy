# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Test PCA.plot_score labeling workflow."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pca_dataset():
    """Small deterministic dataset for PCA plotting tests (20×8, labeled)."""
    rng = np.random.RandomState(42)
    data = rng.randn(20, 8)
    data[:10] += 2.0
    from spectrochempy import Coord
    from spectrochempy import NDDataset

    y = Coord(
        data=np.arange(20),
        labels=np.column_stack(
            [
                np.array([f"R{i}" for i in range(20)]),
                np.array([f"G{i // 5}" for i in range(20)]),
            ]
        ),
    )
    x = Coord(data=np.arange(8))
    return NDDataset(data, coordset=[y, x])


def _precomputed_scores():
    """Precomputed synthetic scores NDDataset (20×5, labeled) for plotting tests."""
    rng = np.random.RandomState(42)
    data = rng.randn(20, 5)
    from spectrochempy import Coord
    from spectrochempy import NDDataset

    y = Coord(
        data=np.arange(20),
        labels=np.column_stack(
            [
                np.array([f"R{i}" for i in range(20)]),
                np.array([f"G{i // 5}" for i in range(20)]),
            ]
        ),
    )
    x = Coord(data=np.arange(5))
    return NDDataset(data, coordset=[y, x])


class TestPlotScoreLabelsWorkflow:
    """Tests for PCA.plot_score labeling workflow."""

    def test_plot_score_labels_workflow(self):
        """
        Test that user-modified scores labels are used in plot_score.

        This tests the workflow:
        1. Obtain scores (precomputed)
        2. Modify scores.y.labels
        3. Call plot_score(scores=scores, show_labels=True)
        4. Verify labels are displayed

        Note: When setting scores.y.labels, the new labels are APPENDED to existing
        labels (Coord behavior), not replaced. The custom labels end up in the last
        column, so labels_column should point to that column.
        """
        from spectrochempy.plotting.composite.plotscore import plot_score

        scores = _precomputed_scores()

        labels = [lab[:6] for lab in scores.y.labels[:, 1]]
        scores.y.labels = labels

        # Custom labels are appended, so use the last column (index 2)
        ax = plot_score(
            scores, components=(1, 2), show_labels=True, labels_column=2, show=False
        )

        assert (
            len(ax.texts) == scores.shape[0]
        ), f"Expected {scores.shape[0]} label texts, got {len(ax.texts)}"
        plt.close()

    def test_plot_score_uses_modified_scores(self):
        """
        Test that plot_score uses passed scores object, not regenerated scores.

        Note: When setting scores.y.labels, the new labels are APPENDED to existing
        labels (Coord behavior), not replaced. The custom labels end up in the last
        column, so labels_column should point to that column.
        """
        from spectrochempy.plotting.composite.plotscore import plot_score

        scores = _precomputed_scores()

        labels = [f"Sample_{i}" for i in range(scores.shape[0])]
        scores.y.labels = labels

        # Custom labels are appended, so use the last column
        custom_labels_column = scores.y.labels.shape[1] - 1

        ax = plot_score(
            scores,
            components=(1, 2),
            show_labels=True,
            labels_column=custom_labels_column,
            show=False,
        )

        text_labels = [t.get_text() for t in ax.texts]
        assert (
            "Sample_0" in text_labels
        ), f"Expected 'Sample_0' in labels, got {text_labels}"
        assert (
            "Sample_5" in text_labels
        ), f"Expected 'Sample_5' in labels, got {text_labels}"
        plt.close()

    def test_plot_score_without_scores_argument(self):
        """Test that plot_score works without passing scores (uses self.scores)."""
        import spectrochempy as scp

        dataset = _pca_dataset()

        pca = scp.PCA(n_components=5)
        pca.fit(dataset)

        ax = pca.plot_score(show=False)

        assert ax is not None
        assert len(ax.collections) == 1
        plt.close()

    def test_scoreplot_backward_compat_with_scores(self):
        """Test backward compatibility: scoreplot(scores, 1, 2) still works."""
        import spectrochempy as scp

        dataset = _pca_dataset()

        pca = scp.PCA(n_components=5)
        pca.fit(dataset)

        scores = pca.transform()

        with pytest.warns(DeprecationWarning):
            ax = pca.scoreplot(scores, 1, 2, show=False)

        assert ax is not None
        plt.close()

    def test_plot_score_components_as_positional(self):
        """Test that passing components as first positional still works."""
        import spectrochempy as scp

        dataset = _pca_dataset()

        pca = scp.PCA(n_components=5)
        pca.fit(dataset)

        ax = pca.plot_score((1, 2), show=False)

        assert ax is not None
        plt.close()

    def test_plot_score_3d(self):
        """
        Test 3D score plot with custom labels.

        Note: Custom labels are appended to existing labels, so use the last column.
        """
        from spectrochempy.plotting.composite.plotscore import plot_score

        scores = _precomputed_scores()

        labels = [f"S{i}" for i in range(scores.shape[0])]
        scores.y.labels = labels

        # Custom labels are appended, so use the last column
        custom_labels_column = scores.y.labels.shape[1] - 1

        ax = plot_score(
            scores,
            components=(1, 2, 3),
            show_labels=True,
            labels_column=custom_labels_column,
            show=False,
        )

        assert ax is not None
        plt.close()
