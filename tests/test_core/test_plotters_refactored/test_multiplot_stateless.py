# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Multiplot stateless behavior tests.

Tests multiplot functionality ensuring proper grid layouts,
return types, and dataset statelessness.
"""

import matplotlib.pyplot as plt
import numpy as np

from spectrochempy.core.plotters.multiplot import multiplot


class TestMultiplotStateless:
    """Test multiplot functionality in stateless architecture."""

    def test_basic_grid_layout(self, sample_1d_dataset):
        """Test 10: Basic multiplot grid layout (2x2)."""
        # Create multiple datasets
        datasets = [sample_1d_dataset] * 4

        # Store original states
        original_states = [ds.__dict__.copy() for ds in datasets]

        # Test 2x2 grid
        axes = multiplot(datasets, nrows=2, ncols=2)

        # Verify return type and structure
        assert isinstance(axes, np.ndarray), "multiplot should return numpy array"
        assert axes.shape == (4,), "Should return array of 4 axes"
        assert all(
            isinstance(ax, plt.Axes) for ax in axes
        ), "All elements should be Axes objects"

        # Verify grid layout (check positions)
        fig = axes[0].figure
        assert len(fig.axes) >= 4, "Figure should have at least 4 subplot axes"

        # Verify datasets unchanged
        for i, (orig_state, dataset) in enumerate(
            zip(original_states, datasets, strict=False)
        ):
            assert_dataset_state_unchanged(orig_state, dataset)

    def test_multiplot_single_dataset(self, sample_1d_dataset):
        """Test 11: Multiplot with single dataset (1x1)."""
        ds_before = sample_1d_dataset.__dict__.copy()

        # Test single dataset multiplot
        axes = multiplot(sample_1d_dataset, nrows=1, ncols=1)

        # Verify return type for single dataset
        assert isinstance(
            axes, plt.Axes
        ), "Single dataset multiplot should return single Axes"

        # Compare with direct plot
        ax_direct = sample_1d_dataset.plot()

        # Should have similar basic properties (both should be line plots)
        assert len(axes.get_lines()) == len(
            ax_direct.get_lines()
        ), "Both should have same number of line objects"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_multiplot_method_selection(self, sample_1d_dataset, sample_2d_dataset):
        """Test 12: Multiplot with mixed method selection."""
        datasets = [sample_1d_dataset, sample_2d_dataset]

        # Store original states
        ds1_before = sample_1d_dataset.__dict__.copy()
        ds2_before = sample_2d_dataset.__dict__.copy()

        # Test with explicit methods
        axes = multiplot(datasets, nrows=1, ncols=2, method=["pen", "map"])

        # Verify return structure
        assert isinstance(axes, np.ndarray), "Should return numpy array"
        assert axes.shape == (2,), "Should return array of 2 axes"

        # Verify method application through basic checks
        # First axes (pen) should have lines
        assert len(axes[0].get_lines()) > 0, "Pen method should create lines"

        # Second axes (map) should have collections for contours
        assert (
            len(axes[1].collections) > 0
        ), "Map method should create contour collections"

        # Verify datasets unchanged
        assert_dataset_state_unchanged(ds1_before, sample_1d_dataset)
        assert_dataset_state_unchanged(ds2_before, sample_2d_dataset)

    def test_multiplot_suptitle(self, sample_1d_dataset):
        """Test 13: Multiplot suptitle handling."""
        datasets = [sample_1d_dataset] * 4
        ds_before = sample_1d_dataset.__dict__.copy()

        # Test with suptitle
        axes = multiplot(datasets, nrows=2, ncols=2, suptitle="Main Test Title")

        # Verify suptitle applied
        fig = axes[0].figure
        suptitle_obj = fig._suptitle if hasattr(fig, "_suptitle") else None

        # Check if title exists somewhere in figure
        found_title = False
        if hasattr(fig, "_suptitle") and fig._suptitle:
            found_title = True
        elif hasattr(fig, "texts"):
            for text in fig.texts:
                if "Main Test Title" in text.get_text():
                    found_title = True
                    break

        assert found_title, "Suptitle should be applied to figure"

        # Verify datasets unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)
