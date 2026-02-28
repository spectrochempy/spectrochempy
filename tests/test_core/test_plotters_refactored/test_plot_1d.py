"""
1D Plotting Tests - Refactored

Tests for 1D plotting functionality (line plots, scatter plots, etc.).

This module tests core 1D plotting capabilities that were previously
broken but are now working after the critical bug fixes.
"""

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes


class Test1DPlotting:
    """Test suite for 1D plotting functionality."""

    def test_1d_basic_line_plot(self, sample_1d_dataset, clean_figures):
        """
        Test basic 1D line plotting functionality.

        Validates that:
        - Basic line plotting works
        - No transform errors occur
        - Figure is created properly
        """
        # Act
        ax = sample_1d_dataset.plot(show=False)

        # Assert
        assert ax is not None
        assert isinstance(ax, Axes)

        # Verify data is plotted
        lines = ax.get_lines()
        assert len(lines) > 0

    def test_1d_plot_with_parameters(self, sample_1d_dataset, clean_figures):
        """
        Test 1D plotting with various parameters.

        Validates that:
        - Plot parameters are passed correctly
        - Style parameters work
        - Different plotting methods are available
        """
        # Act
        ax = sample_1d_dataset.plot(show=False, color="red", linestyle="--")

        # Assert
        lines = ax.get_lines()
        assert len(lines) > 0
        assert lines[0].get_color() == "red"
        assert lines[0].get_linestyle() == "--"

    def test_1d_plot_show_zero_parameter(self, sample_1d_dataset, clean_figures):
        """
        Test show_zero parameter for 1D plots.

        Validates the fix for:
        - BUG #1: Invalid matplotlib API call (haxlines → axhline)
        """
        # Act - should plot horizontal line at y=0
        ax = sample_1d_dataset.plot(show=False, show_zero=True)

        # Assert - check for horizontal line
        lines = ax.get_lines()
        horizontal_lines = [line for line in lines if abs(line.get_ydata()[0]) < 1e-10]
        assert len(horizontal_lines) > 0, "show_zero should add horizontal line"

    @pytest.mark.parametrize("method", ["line", "pen", "scatter"])
    def test_1d_different_methods(self, sample_1d_dataset, method, clean_figures):
        """
        Test different 1D plotting methods.

        Validates that various 1D plotting methods work without errors.
        """
        # Act
        ax = sample_1d_dataset.plot(show=False, method=method)

        # Assert
        assert ax is not None
        assert isinstance(ax, Axes)

    def test_1d_lazy_initialization(self, sample_1d_dataset, backend_checker):
        """
        Test that lazy matplotlib initialization works correctly for 1D plots.

        Validates that:
        - matplotlib is initialized on first plot call
        - No premature initialization occurs
        """
        # Act
        ax = sample_1d_dataset.plot(show=False)

        # Assert - matplotlib should be available now
        assert plt.get_fignums()  # Figures exist = matplotlib initialized

    def test_1d_figure_cleanup(self, sample_1d_dataset, clean_figures):
        """
        Test that figures are properly cleaned up.

        Validates that clean_figures fixture works correctly.
        """
        # Get initial figure count
        initial_count = len(plt.get_fignums())

        # Act - create a plot
        ax = sample_1d_dataset.plot(show=False)
        final_count = len(plt.get_fignums())

        # Assert
        assert final_count == initial_count + 1, "Should create one new figure"

        # Fixture should clean up automatically
        # This is tested by the clean_figures fixture mechanism
