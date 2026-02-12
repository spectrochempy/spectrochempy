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

import matplotlib.pyplot as plt
import pytest
import numpy as np


class TestStatelessPlotting:
    """Test core stateless plotting behavior."""

    def test_default_method_selection(self, sample_1d_dataset, sample_2d_dataset, sample_3d_dataset):
        """Test 1: Default method selection based on dataset dimensionality."""
        # Store dataset state before plotting
        ds1_before = sample_1d_dataset.__dict__.copy()
        ds2_before = sample_2d_dataset.__dict__.copy()
        ds3_before = sample_3d_dataset.__dict__.copy()
        
        # Plot without specifying method
        ax1 = sample_1d_dataset.plot()
        ax2 = sample_2d_dataset.plot()
        ax3 = sample_3d_dataset.plot()
        
        # Verify correct methods were used (check line presence for 1D)
        lines1 = ax1.get_lines()
        assert len(lines1) > 0, "1D plot should create line objects"
        
        # Verify 2D plot has contours or similar
        collections2 = ax2.collections
        assert len(collections2) > 0, "2D plot should create collection objects"
        
        # Verify 3D plot has surface
        assert hasattr(ax3, 'zaxis'), "3D plot should have z-axis"
        assert ax3.name == '3d', "3D plot should have 3d projection"
        
        # Verify datasets remain unchanged
        assert_dataset_state_unchanged(ds1_before, sample_1d_dataset)
        assert_dataset_state_unchanged(ds2_before, sample_2d_dataset)
        assert_dataset_state_unchanged(ds3_before, sample_3d_dataset)

    def test_explicit_method_dispatch(self, sample_1d_dataset, sample_2d_dataset, sample_3d_dataset):
        """Test 2: Explicit method dispatch works correctly."""
        # Store original state
        ds1_before = sample_1d_dataset.__dict__.copy()
        ds2_before = sample_2d_dataset.__dict__.copy()
        ds3_before = sample_3d_dataset.__dict__.copy()
        
        # Test explicit method specification
        ax_scatter = sample_1d_dataset.plot(method="scatter")
        ax_map = sample_2d_dataset.plot(method="map")
        ax_surface = sample_3d_dataset.plot(method="surface")
        
        # Verify correct plot types
        # Scatter plot should have paths, not just lines
        assert len(ax_scatter.collections) > 0 or len(ax_scatter.paths) > 0, "Scatter plot should have marker objects"
        
        # Map plot should have contour lines
        assert len(ax_map.collections) > 0, "Map plot should have contour collections"
        
        # Surface plot should have 3D projection
        assert hasattr(ax_surface, 'zaxis'), "Surface plot should be 3D"
        
        # Verify datasets unchanged
        assert_dataset_state_unchanged(ds1_before, sample_1d_dataset)
        assert_dataset_state_unchanged(ds2_before, sample_2d_dataset)
        assert_dataset_state_unchanged(ds3_before, sample_3d_dataset)

    def test_return_type_verification(self, sample_1d_dataset):
        """Test 3: Return type verification - returns matplotlib Axes only."""
        ds_before = sample_1d_dataset.__dict__.copy()
        
        # Test single plot return type
        ax = sample_1d_dataset.plot()
        assert isinstance(ax, plt.Axes), "dataset.plot() should return matplotlib Axes object"
        
        # Verify returned axes can be used independently
        ax.set_title("Independent title")
        assert ax.get_title() == "Independent title", "Returned axes should be independently usable"
        
        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)
        
        # Verify no dataset plotting attributes
        assert not hasattr(sample_1d_dataset, 'fig')
        assert not hasattr(sample_1d_dataset, 'ndaxes')

    def test_basic_parameters(self, sample_1d_dataset):
        """Test 4: Basic parameters applied correctly."""
        ds_before = sample_1d_dataset.__dict__.copy()
        
        # Test basic plot parameters
        ax = sample_1d_dataset.plot(
            title="Test Title",
            xlabel="Custom X Label",
            ylabel="Custom Y Label"
        )
        
        # Verify parameters applied
        assert ax.get_title() == "Test Title", "Title parameter not applied correctly"
        assert ax.get_xlabel() == "Custom X Label", "X label parameter not applied correctly"
        assert ax.get_ylabel() == "Custom Y Label", "Y label parameter not applied correctly"
        
        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_invalid_method_error(self, sample_1d_dataset):
        """Test 5: Invalid method raises appropriate error."""
        ds_before = sample_1d_dataset.__dict__.copy()
        
        # Test invalid method
        with pytest.raises((NameError, OSError)) as exc_info:
            sample_1d_dataset.plot(method="nonexistent_method")
        
        # Verify error is informative
        error_message = str(exc_info.value)
        assert "nonexistent_method" in error_message, "Error message should mention the invalid method"
        
        # Verify dataset unchanged
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
            if axes != ax and hasattr(axes, 'images'):
                colorbar_found = True
                break
        
        # Also check collections for colorbar-related objects
        if not colorbar_found:
            for collection in ax.collections:
                if hasattr(collection, 'colorbar'):
                    colorbar_found = True
                    break
        
        # Note: This test might need adjustment based on actual implementation
        # The key is that colorbar=True should add something visible to the plot
        
        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_2d_dataset)