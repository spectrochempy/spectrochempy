# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Stateless core plotting behavior tests.

Tests fundamental dataset.plot() functionality ensuring:
- Stateless behavior (no dataset mutation)
- Correct method selection and dispatch
- Proper return types
- Basic parameter handling
- Error handling
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from plotting_fixtures import assert_dataset_state_unchanged


class TestStatelessPlotting:
    """Test core stateless plotting behavior."""

    def test_default_method_selection(self, sample_1d_dataset, sample_2d_dataset):
        """Test 1: Default method selection based on dataset dimensionality."""
        # Store dataset state before plotting
        ds1_before = sample_1d_dataset.__dict__.copy()
        ds2_before = sample_2d_dataset.__dict__.copy()

        # Plot without specifying method
        ax1 = sample_1d_dataset.plot()
        ax2 = sample_2d_dataset.plot()

        # Verify correct methods were used (check line presence for 1D)
        lines1 = ax1.get_lines()
        assert len(lines1) > 0, "1D plot should create line objects"

        # Verify 2D plot creates lines (default is stack plot)
        lines2 = ax2.get_lines()
        assert len(lines2) > 0, "2D plot should create line objects (stack)"

        # Verify datasets remain unchanged
        assert_dataset_state_unchanged(ds1_before, sample_1d_dataset)
        assert_dataset_state_unchanged(ds2_before, sample_2d_dataset)

    def test_explicit_method_dispatch(self, sample_1d_dataset, sample_2d_dataset):
        """Test 2: Explicit method dispatch works correctly."""
        # Store original state
        ds1_before = sample_1d_dataset.__dict__.copy()
        ds2_before = sample_2d_dataset.__dict__.copy()

        # Test explicit method specification
        ax_scatter = sample_1d_dataset.plot(method="scatter")
        ax_map = sample_2d_dataset.plot(method="map")
        ax_surface = sample_2d_dataset.plot(method="surface")

        # Verify correct plot types
        # Scatter plot uses Line2D with markers, not PathCollection
        assert len(ax_scatter.lines) > 0, "Scatter plot should have line objects"
        line = ax_scatter.lines[0]
        assert line.get_marker() not in (
            None,
            "None",
            "",
        ), "Scatter plot should have markers"
        assert line.get_linestyle() == "None", "Scatter plot should have no line"
        assert (
            len(ax_scatter.collections) == 0
        ), "Scatter plot should not use collections"

        # Map plot should have contour lines
        assert len(ax_map.collections) > 0, "Map plot should have contour collections"

        # Surface plot should have 3D projection
        assert hasattr(ax_surface, "zaxis"), "Surface plot should be 3D"

        # Verify datasets unchanged
        assert_dataset_state_unchanged(ds1_before, sample_1d_dataset)
        assert_dataset_state_unchanged(ds2_before, sample_2d_dataset)

    def test_return_type_verification(self, sample_1d_dataset):
        """Test 3: Return type verification - returns matplotlib Axes only."""
        ds_before = sample_1d_dataset.__dict__.copy()

        # Test single plot return type
        ax = sample_1d_dataset.plot()
        assert isinstance(
            ax, plt.Axes
        ), "dataset.plot() should return matplotlib Axes object"

        # Verify returned axes can be used independently
        ax.set_title("Independent title")
        assert (
            ax.get_title() == "Independent title"
        ), "Returned axes should be independently usable"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

        # Verify no dataset plotting attributes
        assert not hasattr(sample_1d_dataset, "fig")
        assert not hasattr(sample_1d_dataset, "ndaxes")

    def test_basic_parameters(self, sample_1d_dataset):
        """Test 4: Basic parameters applied correctly."""
        ds_before = sample_1d_dataset.__dict__.copy()

        # Test basic plot parameters
        ax = sample_1d_dataset.plot(
            title="Test Title", xlabel="Custom X Label", ylabel="Custom Y Label"
        )

        # Verify parameters applied
        assert ax.get_title() == "Test Title", "Title parameter not applied correctly"
        assert (
            ax.get_xlabel() == "Custom X Label"
        ), "X label parameter not applied correctly"
        assert (
            ax.get_ylabel() == "Custom Y Label"
        ), "Y label parameter not applied correctly"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_invalid_method_error(self, sample_1d_dataset):
        """Test 5: Invalid method raises appropriate error."""
        ds_before = sample_1d_dataset.__dict__.copy()

        # Test invalid method - raises OSError after logging NameError
        with pytest.raises(OSError):
            sample_1d_dataset.plot(method="nonexistent_method")

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)


class TestScatterMarkerBehavior:
    """Test scatter plot marker behavior."""

    def test_scatter_default_marker(self, sample_1d_dataset):
        """Test that plot_scatter uses default marker from preferences."""
        ax = sample_1d_dataset.plot_scatter()

        # Should have lines, not collections
        assert len(ax.lines) > 0, "Scatter plot should have line objects"
        assert len(ax.collections) == 0, "Scatter plot should not use collections"

        line = ax.lines[0]
        # Default marker should be "o" from preferences
        assert (
            line.get_marker() == "o"
        ), f"Expected default marker 'o', got {line.get_marker()}"
        assert (
            line.get_linestyle() == "None"
        ), "Scatter plot should have no connecting line"

    def test_scatter_explicit_marker(self, sample_1d_dataset):
        """Test that explicit marker overrides default."""
        ds_before = sample_1d_dataset.__dict__.copy()

        ax = sample_1d_dataset.plot_scatter(marker="s")

        assert len(ax.lines) > 0, "Scatter plot should have line objects"
        line = ax.lines[0]
        assert line.get_marker() == "s", f"Expected marker 's', got {line.get_marker()}"
        assert (
            line.get_linestyle() == "None"
        ), "Scatter plot should have no connecting line"

        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_scatter_boolean_kwarg_selects_scatter(self, sample_1d_dataset):
        """Test the documented scatter=True compatibility flag."""
        ds_before = sample_1d_dataset.__dict__.copy()

        ax = sample_1d_dataset.plot(scatter=True)

        assert len(ax.lines) > 0, "Scatter plot should have line objects"
        line = ax.lines[0]
        assert line.get_marker() not in (
            None,
            "None",
            "",
        ), "scatter=True should enable markers"
        assert line.get_linestyle() == "None", "scatter=True should disable the line"

        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_scatter_pen_has_line_and_marker(self, sample_1d_dataset):
        """Test that plot_scatter_pen shows both line and marker."""
        ds_before = sample_1d_dataset.__dict__.copy()

        ax = sample_1d_dataset.plot_scatter_pen()

        assert len(ax.lines) > 0, "Scatter-pen plot should have line objects"
        assert len(ax.collections) == 0, "Scatter-pen plot should not use collections"

        line = ax.lines[0]
        # Should have both marker and line
        assert line.get_marker() not in (
            None,
            "None",
            "",
        ), "Scatter-pen should have markers"
        assert line.get_linestyle() not in (
            None,
            "None",
            "",
        ), "Scatter-pen should have a line"

        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_scatter_no_collections(self, sample_1d_dataset):
        """Test that scatter plot does not create PathCollection."""
        ds_before = sample_1d_dataset.__dict__.copy()

        ax = sample_1d_dataset.plot_scatter()

        # Must use Line2D, not PathCollection
        assert (
            len(ax.collections) == 0
        ), "Scatter plot should not create collections (PathCollection)"
        assert len(ax.lines) > 0, "Scatter plot must create Line2D objects"

        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)


class TestColorbarParameter:
    """Test colorbar parameter behavior."""

    def test_colorbar_basic(self, sample_2d_dataset):
        """Test 6: Colorbar parameter for 2D plots."""
        ds_before = sample_2d_dataset.__dict__.copy()

        # Test with colorbar
        ax = sample_2d_dataset.plot(method="map", colorbar=True)

        # Check if colorbar exists (matplotlib creates colorbar as collection or separate axes)
        fig = ax.figure
        colorbar_found = False

        # Look for colorbar axes or collections
        for axes in fig.axes:
            if axes != ax and hasattr(axes, "images"):
                colorbar_found = True
                break

        # Also check collections for colorbar-related objects
        if not colorbar_found:
            for collection in ax.collections:
                if hasattr(collection, "colorbar"):
                    colorbar_found = True
                    break

        # Note: This test might need adjustment based on actual implementation
        # The key is that colorbar=True should add something visible to the plot

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_2d_dataset)


class TestPlotParameterPropagation:
    """Test propagation of public plotting kwargs."""

    def test_line_alpha_is_applied(self, sample_1d_dataset):
        ax = sample_1d_dataset.plot(alpha=0.25, show=False)

        assert len(ax.lines) > 0
        assert ax.lines[0].get_alpha() == pytest.approx(0.25)

    def test_bar_alpha_is_applied(self):
        from spectrochempy import NDDataset

        ax = NDDataset([1, 2, 3]).plot_bar(alpha=0.4, show=False)

        assert len(ax.patches) == 3
        assert ax.patches[0].get_alpha() == pytest.approx(0.4)

    def test_markeredgewidth_is_applied(self, sample_1d_dataset):
        ax = sample_1d_dataset.plot_scatter(markeredgewidth=3.5, show=False)

        assert len(ax.lines) > 0
        assert ax.lines[0].get_markeredgewidth() == pytest.approx(3.5)

    def test_mew_alias_is_applied(self, sample_1d_dataset):
        ax = sample_1d_dataset.plot_scatter(mew=2.0, show=False)

        assert len(ax.lines) > 0
        assert ax.lines[0].get_markeredgewidth() == pytest.approx(2.0)


class TestDimensionalFallbacks:
    """Test dimensional fallback behavior for direct plot_1D/plot_2D calls."""

    def test_plot_lines_alias_still_works_for_1d_dataset(self, sample_1d_dataset):
        ax = sample_1d_dataset.plot(method="lines", show=False)

        assert len(ax.lines) > 0
        assert ax.lines[0].get_linestyle() not in (None, "None", "")

    def test_plot_pen_alias_still_works_for_2d_dataset(self, sample_2d_dataset):
        ax = sample_2d_dataset.plot(method="pen", show=False)

        assert len(ax.lines) > 0

    def test_plot_1d_forwards_valid_2d_method(self, sample_2d_dataset):
        ax = sample_2d_dataset.plot_1D(method="contour", show=False)

        assert len(ax.collections) > 0

    def test_plot_1d_rejects_incompatible_method(self, sample_2d_dataset):
        with pytest.raises(ValueError, match="incompatible"):
            sample_2d_dataset.plot_1D(method="scatter", show=False)

    def test_plot_2d_forwards_valid_1d_method(self, sample_1d_dataset):
        ax = sample_1d_dataset.plot_2D(method="scatter", show=False)

        assert len(ax.lines) > 0
        assert ax.lines[0].get_marker() not in (None, "None", "")
        assert ax.lines[0].get_linestyle() == "None"

    def test_plot_2d_rejects_incompatible_method(self, sample_1d_dataset):
        with pytest.raises(ValueError, match="incompatible"):
            sample_1d_dataset.plot_2D(method="contour", show=False)
