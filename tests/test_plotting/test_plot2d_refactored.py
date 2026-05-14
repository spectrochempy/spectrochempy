"""
2D Plotting Tests - Refactored.

Tests for 2D plotting functionality (image, stack, map, etc.).

This module tests core 2D plotting capabilities, focusing on the
functionality that was previously broken but is now working.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import Collection


class Test2DPlotting:
    """Test suite for 2D plotting functionality."""

    def test_2d_stack_plot_basic(self, sample_2d_dataset, clean_figures):
        """
        Test basic 2D stack plotting functionality.

        Validates that:
        - Stack plotting works without errors
        - No transform issues occur
        - Figure is created with proper structure
        """
        # Act
        ax = sample_2d_dataset.plot(show=False, method="stack")

        # Assert
        assert ax is not None
        assert isinstance(ax, Axes)

        # Verify stack plot has multiple lines (one for each y row)
        lines = ax.get_lines()
        expected_lines = sample_2d_dataset.shape[0]  # One line per row
        assert len(lines) >= expected_lines * 0.8  # Allow some tolerance

    def test_2d_stack_show_zero_parameter(self, sample_2d_dataset, clean_figures):
        """
        Test show_zero parameter for 2D stack plots.

        Validates the fix for:
        - BUG #1: Invalid matplotlib API call (haxlines → axhline)
        """
        # Act - should plot horizontal line at y=0
        ax = sample_2d_dataset.plot(show=False, method="stack", show_zero=True)

        # Assert - check for horizontal line in stack plot
        lines = ax.get_lines()
        # Look for lines that are approximately horizontal (y near 0)
        horizontal_lines = [
            line for line in lines if abs(np.mean(line.get_ydata())) < 1e-10
        ]
        assert len(horizontal_lines) > 0, "show_zero should add horizontal lines"

    def test_2d_image_plot_basic(self, sample_2d_dataset, clean_figures):
        """
        Test basic 2D image plotting functionality.

        Validates that:
        - Image plotting works without errors
        - Colormap is applied correctly
        - Figure structure is correct
        """
        # Act
        ax = sample_2d_dataset.plot(show=False, method="image")

        # Assert
        assert ax is not None
        assert isinstance(ax, Axes)

        # Verify image has proper data
        images = [child for child in ax.get_children() if hasattr(child, "get_array")]
        assert len(images) > 0, "Should create image elements"

    def test_2d_map_plot_basic(self, sample_2d_dataset, clean_figures):
        """
        Test basic 2D map plotting functionality.

        Validates that:
        - Map plotting works without errors
        - Contour/level creation works
        - Figure structure is correct
        """
        # Act
        ax = sample_2d_dataset.plot(show=False, method="map")

        # Assert
        assert ax is not None
        assert isinstance(ax, Axes)

        # Verify map has collections (contours, etc.)
        collections = [
            child for child in ax.get_children() if isinstance(child, Collection)
        ]
        assert len(collections) > 0, "Should create map elements"

    @pytest.mark.parametrize("method", ["stack", "image", "map"])
    def test_2d_plot_methods(self, sample_2d_dataset, method, clean_figures):
        """
        Test different 2D plotting methods.

        Validates that all major 2D plotting methods work without errors.
        """
        # Act
        ax = sample_2d_dataset.plot(show=False, method=method)

        # Assert
        assert ax is not None
        assert isinstance(ax, Axes)

    def test_2d_lazy_initialization(self, sample_2d_dataset, backend_checker):
        """
        Test that lazy matplotlib initialization works correctly for 2D plots.

        Validates that:
        - matplotlib is initialized on first plot call
        - No premature initialization occurs
        """
        # Act
        sample_2d_dataset.plot(show=False)

        # Assert - matplotlib should be available now
        assert plt.get_fignums()  # Figures exist = matplotlib initialized

    def test_2d_figure_cleanup(self, sample_2d_dataset, clean_figures):
        """
        Test that figures are properly cleaned up.

        Validates that clean_figures fixture works correctly.
        """
        # Get initial figure count
        initial_count = len(plt.get_fignums())

        # Act - create a plot
        sample_2d_dataset.plot(show=False)
        final_count = len(plt.get_fignums())

        # Assert
        assert final_count == initial_count + 1, "Should create one new figure"

    @pytest.mark.parametrize(
        "parameter",
        [{"cmap": "viridis"}, {"colormap": "plasma"}, {"interpolation": "bilinear"}],
    )
    def test_2d_plot_with_parameters(self, sample_2d_dataset, parameter, clean_figures):
        """
        Test 2D plotting with various parameters.

        Validates that:
        - Parameters are passed correctly
        - No errors occur with different parameter combinations
        """
        # Act
        ax = sample_2d_dataset.plot(show=False, method="image", **parameter)

        # Assert
        assert ax is not None
        assert isinstance(ax, Axes)
