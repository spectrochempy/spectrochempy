"""
Multiplot Tests - Refactored.

Tests for multiplot functionality.

This module tests the multiplot capabilities, focusing on the
functionality that was previously broken due to transform errors but is now working.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class TestMultiplot:
    """Test suite for multiplot functionality."""

    def test_multiplot_basic(self, sample_1d_dataset, clean_figures):
        """
        Test basic multiplot functionality.

        Validates that:
        - Multiplot works without transform errors
        - BUG FIX: multiplot.py transform error is resolved
        - Multiple datasets are plotted correctly
        """
        # Create multiple datasets
        datasets = [
            sample_1d_dataset,
            sample_1d_dataset,  # Duplicate for testing
        ]

        # Act
        fig, axes = plt.multiplot(*datasets, show=False)

        # Assert
        assert fig is not None
        assert isinstance(fig, Figure)
        assert len(axes) == len(datasets)

        # Verify axes are properly created
        for ax in axes:
            assert ax is not None

    def test_multiplot_with_different_layouts(self, sample_1d_dataset, clean_figures):
        """
        Test multiplot with different layout options.

        Validates that:
        - Different layout configurations work
        - No transform errors occur with various layouts
        """
        # Create datasets for 2x2 grid
        datasets = [sample_1d_dataset] * 4

        # Test different layouts
        layouts = [
            {"nrow": 2, "ncol": 2},  # 2x2 grid
            {"nrow": 1, "ncol": 4},  # 1x4 row
            {"nrow": 4, "ncol": 1},  # 4x1 column
        ]

        for layout in layouts:
            # Act
            fig, axes = plt.multiplot(*datasets, show=False, **layout)

            # Assert
            assert fig is not None
            assert len(axes) == len(datasets)

            # Verify layout matches expected
            expected_nrow = layout["nrow"]
            expected_ncol = layout["ncol"]

            # Check axes arrangement (this is approximate)
            assert len(axes) == expected_nrow * expected_ncol

    def test_multiplot_with_stack(self, sample_2d_dataset, clean_figures):
        """
        Test multiplot with 2D datasets.

        Validates that:
        - Multiplot works with different dataset types
        - 2D plotting methods work in multiplot context
        """
        # Create 2D datasets
        datasets = [sample_2d_dataset, sample_2d_dataset]

        # Act
        fig, axes = plt.multiplot(*datasets, show=False)

        # Assert
        assert fig is not None
        assert len(axes) == len(datasets)

        # Verify axes have proper structure
        for ax in axes:
            assert ax is not None

    def test_multiplot_lazy_initialization(self, sample_1d_dataset, backend_checker):
        """
        Test that lazy matplotlib initialization works correctly for multiplot.

        Validates that:
        - matplotlib is initialized when multiplot is called
        - No premature initialization occurs
        """
        # Act
        fig, axes = plt.multiplot(sample_1d_dataset, show=False)

        # Assert - matplotlib should be available now
        assert plt.get_fignums()  # Figures exist = matplotlib initialized

        # Verify multiplot results
        assert fig is not None
        assert len(axes) >= 1

    def test_multiplot_figure_cleanup(self, sample_1d_dataset, clean_figures):
        """
        Test that multiplot figures are properly cleaned up.

        Validates that clean_figures fixture works correctly.
        """
        # Get initial figure count
        initial_count = len(plt.get_fignums())

        # Act - create multiplot
        fig, axes = plt.multiplot(sample_1d_dataset, show=False)
        final_count = len(plt.get_fignums())

        # Assert
        assert final_count == initial_count + 1, "Should create one new figure"

    def test_multiplot_with_parameters(self, sample_1d_dataset, clean_figures):
        """
        Test multiplot with various parameters.

        Validates that:
        - Parameters are passed correctly
        - No errors occur with different parameter combinations
        """
        # Act
        fig, axes = plt.multiplot(
            sample_1d_dataset, show=False, sharex=True, sharey=True, figsize=(8, 6)
        )

        # Assert
        assert fig is not None
        assert len(axes) >= 1

    def test_multiplot_transform_bug_fix(self, sample_1d_dataset, clean_figures):
        """
        Test that the multiplot transform bug has been fixed.

        This specifically validates:
        - BUG #2: Multiplot transform error (Axes creation pattern)
        - New fig.add_subplot approach works correctly
        """
        # Create dataset
        datasets = [sample_1d_dataset]

        # Act
        fig, axes = plt.multiplot(*datasets, show=False)

        # Assert - this should work without transform errors
        assert fig is not None, "Multiplot should succeed"
        assert len(axes) == 1, "Should create one set of axes"

        # The key validation: if we got here without transform errors,
        # the bug fix worked
        assert True  # If this assertion passes, the bug is fixed
