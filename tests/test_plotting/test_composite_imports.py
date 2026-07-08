# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Minimal tests for composite plotting modules to improve coverage.

These tests ensure the modules can be imported and basic functions are callable.
"""

import numpy as np


class TestPlotMeritImports:
    """Test that plotmerit module can be imported and has expected functions."""

    def test_plot_merit_import(self):
        """Test import of plot_merit."""
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        assert callable(plot_merit)

    def test_plot_compare_import(self):
        """Test import of plot_compare."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        assert callable(plot_compare)

    def test_plot_compare_makes_reconstructed_trace_visible(self, sample_1d_dataset):
        """The reconstructed profile should remain visually distinct."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        ax = plot_compare(X, X_ref, show=False)

        assert len(ax.lines) == 3
        assert ax.lines[1].get_color() == "tab:blue"
        assert ax.lines[2].get_color() == "tab:orange"
        assert ax.lines[2].get_linestyle() == "--"

    def test_plot_compare_scatter_mode_uses_markers(self, sample_1d_dataset):
        """Scatter mode should render point markers rather than solid lines."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        ax = plot_compare(X, X_ref, kind="scatter", show=False)

        assert len(ax.lines) == 3
        for line in ax.lines:
            assert line.get_marker() == "o"
            assert line.get_linestyle() == "None"

    def test_plot_compare_nb_traces_limits_display(self, sample_2d_dataset):
        """nb_traces should reduce the number of rendered traces."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_2d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        ax = plot_compare(X, X_ref, nb_traces=3, show=False)

        assert len(ax.lines) == 9

    def test_plot_compare_offset_moves_residual_down(self, sample_1d_dataset):
        """Offset should separate the residual from the main traces."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X * 0.9
        X_ref.name = "fit"

        ax = plot_compare(X, X_ref, offset=20, show=False)

        signal_range = max(np.nanmax(X.data), np.nanmax(X_ref.data)) - min(
            np.nanmin(X.data),
            np.nanmin(X_ref.data),
        )
        expected_shift = 0.2 * signal_range
        expected_residual = (X - X_ref).data - expected_shift

        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_residual)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), X.data)
        np.testing.assert_allclose(ax.lines[2].get_ydata(), X_ref.data)

    def test_plot_compare_uses_scpy_figure_size_preference(self, sample_1d_dataset):
        """Standalone comparison figures should follow the current SCPy figure size."""
        from spectrochempy import preferences
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        original_figsize = tuple(preferences.figure.figsize)
        preferences.figure.figsize = (7.0, 3.0)
        try:
            ax = plot_compare(X, X_ref, show=False)
            width, height = ax.figure.get_size_inches()
            np.testing.assert_allclose((width, height), (7.0, 3.0))
            assert ax.figure.get_tight_layout() is True
        finally:
            preferences.figure.figsize = original_figsize

    def test_plot_compare_accepts_short_legend_labels_and_position(
        self,
        sample_1d_dataset,
    ):
        """Legend labels and location should be customizable for compact figures."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        ax = plot_compare(
            X,
            X_ref,
            exp_label="exp",
            calc_label="fit",
            resid_label="res",
            legend_loc="upper left",
            show=False,
        )

        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["exp", "fit", "res"]


class TestMultiplotImports:
    """Test that multiplot module can be imported."""

    def test_multiplot_import(self):
        """Test import of multiplot."""
        from spectrochempy.plotting.multiplot import multiplot

        assert callable(multiplot)

    def test_multiplot_scatter_import(self):
        """Test import of multiplot_scatter."""
        from spectrochempy.plotting.multiplot import multiplot_scatter

        assert callable(multiplot_scatter)

    def test_multiplot_lines_import(self):
        """Test import of multiplot_lines."""
        from spectrochempy.plotting.multiplot import multiplot_lines

        assert callable(multiplot_lines)


class TestProfileImports:
    """Test that profile module can be imported."""

    def test_profile_manager_import(self):
        """Test import of PlotProfileManager."""
        from spectrochempy.plotting.profile import PlotProfileManager

        assert PlotProfileManager is not None

    def test_get_plot_profile_import(self):
        """Test import of get_plot_profile."""
        from spectrochempy.plotting.profile import get_plot_profile

        assert callable(get_plot_profile)

    def test_set_plot_profile_import(self):
        """Test import of set_plot_profile."""
        from spectrochempy.plotting.profile import set_plot_profile

        assert callable(set_plot_profile)
