# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Tests for plot_scree composite function.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestPlotScree:
    """Tests for plot_scree composite function."""

    def test_plot_scree_basic(self):
        """Test basic plot_scree functionality."""
        from spectrochempy.plotting.composite import plot_scree

        explained = np.array([40.0, 25.0, 15.0, 10.0, 5.0, 3.0, 2.0])

        ax = plot_scree(explained, show=False)

        assert ax is not None
        assert len(ax.figure.axes) == 2
        assert len(ax.containers) == 1
        assert len(ax.containers[0]) == len(explained)

        plt.close()

    def test_plot_scree_with_cumulative(self):
        """Test plot_scree with pre-computed cumulative."""
        from spectrochempy.plotting.composite import plot_scree

        explained = np.array([40.0, 25.0, 15.0, 10.0, 5.0])
        cumulative = np.cumsum(explained)

        ax = plot_scree(explained, cumulative=cumulative, show=False)

        twin_axes = [child for child in ax.figure.axes if child != ax]
        assert len(twin_axes) == 1

        plt.close()

    def test_pca_plot_scree_wrapper(self):
        """Test PCA.plot_scree() wrapper."""
        import spectrochempy as scp

        X = scp.read("irdata/nh4y-activation.spg")
        pca = scp.PCA(n_components=5)
        pca.fit(X)

        ax = pca.plot_scree(show=False)

        assert ax is not None

        plt.close()

    def test_pca_screeplot_deprecated(self):
        """Test PCA.screeplot() emits DeprecationWarning."""
        import spectrochempy as scp

        X = scp.read("irdata/nh4y-activation.spg")
        pca = scp.PCA(n_components=5)
        pca.fit(X)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = pca.screeplot(show=False)

            deprecation_warnings = [
                item for item in w if issubclass(item.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

        assert ax is not None

        plt.close()

    def test_plot_scree_deterministic_limits(self):
        """Test that axis limits are deterministic."""
        from spectrochempy.plotting.composite import plot_scree

        explained = np.array([30.0, 20.0, 15.0, 10.0])
        n = len(explained)

        ax = plot_scree(explained, show=False)

        xlim = ax.get_xlim()
        assert xlim[0] == pytest.approx(0.5)
        assert xlim[1] == pytest.approx(n + 0.5)

        ylim = ax.get_ylim()
        assert ylim[0] == pytest.approx(0.0)
        assert ylim[1] == pytest.approx(max(explained) * 1.05)

        right_ax = ax.figure.axes[1]
        right_ylim = right_ax.get_ylim()
        assert right_ylim[1] == 100
        assert right_ylim[0] < explained[0]
        assert right_ylim[0] >= 0

        plt.close()

    def test_plot_scree_custom_colors(self):
        """Test plot_scree with custom colors."""
        from spectrochempy.plotting.composite import plot_scree

        explained = np.array([40.0, 25.0, 15.0])

        ax = plot_scree(explained, show=False, bar_color="red", line_color="green")

        bars = ax.containers[0]
        assert bars[0].get_facecolor()[:3] == pytest.approx((1.0, 0.0, 0.0))

        plt.close()

    def test_plot_scree_with_ax(self):
        """Test plot_scree with provided axes."""
        from spectrochempy.plotting.composite import plot_scree

        explained = np.array([40.0, 25.0, 15.0])

        fig, ax = plt.subplots()
        result = plot_scree(explained, ax=ax, show=False)

        assert result is ax

        plt.close()

    def test_plot_scree_no_title(self):
        """Test plot_scree with title=None."""
        from spectrochempy.plotting.composite import plot_scree

        explained = np.array([40.0, 25.0, 15.0])

        ax = plot_scree(explained, title=None, show=False)

        assert ax.get_title() == ""

        plt.close()
