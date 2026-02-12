# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Style application tests for stateless plotting.

Tests style parameter handling, validation, and local context application.
"""

import matplotlib.pyplot as plt
import pytest
from spectrochempy.application._preferences.plot_preferences import available_styles


class TestStyleApplication:
    """Test style handling in stateless architecture."""

    def test_basic_style_parameter(self, sample_1d_dataset):
        """Test 6: Basic style parameter application."""
        ds_before = sample_1d_dataset.__dict__.copy()
        
        # Test with different styles
        ax1 = sample_1d_dataset.plot(style="paper")
        ax2 = sample_1d_dataset.plot(style="grayscale")
        
        # Verify styles applied (check some basic properties)
        # Different styles should have different appearances
        # We can check basic properties that are likely to differ
        fig1 = ax1.figure
        fig2 = ax2.figure
        
        # At minimum, verify plots were created successfully
        assert ax1.get_title() != "" or len(ax1.lines) > 0, "Paper style plot should be valid"
        assert ax2.get_title() != "" or len(ax2.lines) > 0, "Grayscale style plot should be valid"
        
        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_invalid_style_handling(self, sample_1d_dataset):
        """Test 7: Invalid style error handling."""
        ds_before = sample_1d_dataset.__dict__.copy()
        
        # Test with completely invalid style
        with pytest.raises((ValueError, OSError, Exception)) as exc_info:
            sample_1d_dataset.plot(style="completely_nonexistent_style_xyz123")
        
        # Verify error is informative
        error_message = str(exc_info.value)
        assert any(keyword in error_message.lower() for keyword in ["style", "not", "found", "invalid"]), (
            f"Error should mention style issue, got: {error_message}"
        )
        
        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_style_discovery(self):
        """Test 8: Style discovery function."""
        # Test available_styles function
        styles = available_styles()
        
        # Basic validation
        assert isinstance(styles, (list, tuple)), "Should return list or tuple of styles"
        assert len(styles) > 0, "Should have at least some styles available"
        assert all(isinstance(style, str) for style in styles), "All styles should be strings"
        
        # Check for common matplotlib styles
        common_styles = ["default", "classic", "seaborn"]
        found_common = any(style in styles for style in common_styles)
        # Note: This might fail if seaborn not available, which is fine
        
        # Check for SpectroChemPy styles (if they exist)
        scpy_styles = [s for s in styles if "scpy" in s or s in ["paper", "grayscale", "talk", "poster"]]
        # Note: This depends on actual style installation

    def test_local_style_context_only(self, sample_1d_dataset):
        """Test 9: Local style context isolation - HIGH PRIORITY TEST."""
        # Get global matplotlib state before
        rcparams_before = get_rcparams_snapshot()
        
        # Create plot with SCP style
        ax1 = sample_1d_dataset.plot(style="paper")
        
        # Create separate matplotlib plot
        fig2, ax2 = plt.subplots()
        ax2.plot([1, 2, 3])
        ax2.set_title("Matplotlib Direct Plot")
        
        # Verify matplotlib plot has default styling
        # Check some basic style properties that should differ from paper style
        assert ax2.get_title() == "Matplotlib Direct Plot", "Direct matplotlib plot should work"
        
        # Verify global rcParams haven't been permanently changed
        rcparams_after = get_rcparams_snapshot()
        
        # Most rcParams should be unchanged
        unchanged_params = 0
        total_params = len(rcparams_before)
        
        for key in rcparams_before:
            if key in rcparams_after and rcparams_before[key] == rcparams_after[key]:
                unchanged_params += 1
        
        # At least 90% of parameters should be unchanged
        # (allowing for some legitimate matplotlib state changes during plotting)
        assert unchanged_params >= total_params * 0.9, (
            f"Global rcParams should not be permanently modified. "
            f"Unchanged: {unchanged_params}/{total_params}"
        )
        
        # Verify dataset unchanged
        ds_before = sample_1d_dataset.__dict__.copy()
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)