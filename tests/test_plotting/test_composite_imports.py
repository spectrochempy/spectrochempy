# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Minimal tests for composite plotting modules to improve coverage.

These tests ensure the modules can be imported and basic functions are callable.
"""


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


class TestIrisImports:
    """Test that iris module can be imported and has expected functions."""

    def test_plot_iris_lcurve_import(self):
        """Test import of plot_iris_lcurve."""
        from spectrochempy.plotting.composite.iris import plot_iris_lcurve

        assert callable(plot_iris_lcurve)

    def test_plot_iris_distribution_import(self):
        """Test import of plot_iris_distribution."""
        from spectrochempy.plotting.composite.iris import plot_iris_distribution

        assert callable(plot_iris_distribution)

    def test_plot_iris_merit_import(self):
        """Test import of plot_iris_merit."""
        from spectrochempy.plotting.composite.iris import plot_iris_merit

        assert callable(plot_iris_merit)


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
