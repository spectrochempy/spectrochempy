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
import pytest

from spectrochempy import multiplot


def assert_dataset_state_unchanged(dataset_before, dataset_after):
    """Verify dataset was not mutated by plotting."""
    before_dict = (
        dataset_before if isinstance(dataset_before, dict) else dataset_before.__dict__
    )
    after_dict = dataset_after.__dict__

    # Internal attributes that may be lazily created (not plotting-related)
    internal_attrs = {"_NDDataset__mask_metadata", "__mask_metadata", "_mask_metadata"}

    # Find new keys that aren't internal lazy-init attributes
    new_keys = set(after_dict.keys()) - set(before_dict.keys())
    plotting_keys = new_keys - internal_attrs

    assert (
        not plotting_keys
    ), f"Dataset mutated by plotting with new attributes: {plotting_keys}"
    assert not hasattr(dataset_after, "fig")
    assert not hasattr(dataset_after, "ndaxes")


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
        for orig_state, dataset in zip(original_states, datasets, strict=False):
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

    def test_multiplot_single_dataset_preserves_requested_method(
        self, sample_1d_dataset
    ):
        """Single-dataset multiplot should preserve an explicit method."""
        ax = multiplot(sample_1d_dataset, method="scatter", nrow=1, ncol=1, show=False)

        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) > 0
        line = ax.lines[0]
        assert line.get_marker() not in (None, "None", "")
        assert line.get_linestyle() == "None"

    def test_multiplot_supports_nrows_ncols_aliases(self, sample_1d_dataset):
        """nrows/ncols aliases should behave like nrow/ncol."""
        datasets = [sample_1d_dataset, sample_1d_dataset]

        axes = multiplot(datasets, nrows=1, ncols=2, show=False)

        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2,)

    def test_multiplot_method_selection(self, sample_1d_dataset):
        """Test 12: Multiplot with mixed method selection."""
        # Use two 1D datasets with different methods
        datasets = [sample_1d_dataset, sample_1d_dataset]

        # Store original states
        ds1_before = sample_1d_dataset.__dict__.copy()

        # Test with explicit methods - both 1D using pen and scatter
        axes = multiplot(datasets, nrows=1, ncols=2, method=["pen", "scatter"])

        # Verify return structure
        assert isinstance(axes, np.ndarray), "Should return numpy array"
        assert axes.shape == (2,), "Should return array of 2 axes"

        # Verify method application through basic checks
        # Both should have some content
        assert len(axes) == 2

        # Verify datasets unchanged
        assert_dataset_state_unchanged(ds1_before, sample_1d_dataset)

    def test_multiplot_suptitle(self, sample_1d_dataset):
        """Test 13: Multiplot suptitle handling."""
        datasets = [sample_1d_dataset] * 4
        ds_before = sample_1d_dataset.__dict__.copy()

        # Test with suptitle
        axes = multiplot(datasets, nrows=2, ncols=2, suptitle="Main Test Title")

        # Verify suptitle applied
        fig = axes[0].figure

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


class TestMultiplotGridSize:
    """Grid sizes that do not match the number of datasets (issue #1946)."""

    def test_fewer_datasets_than_panels_hides_unused_axes(self, sample_1d_dataset):
        result = multiplot(
            [sample_1d_dataset] * 3, nrow=2, ncol=2, show=False, return_dict=True
        )

        axes = result["axes"]
        assert axes.shape == (4,)
        assert list(result["axes_dict"]) == ["axe11", "axe12", "axe21", "axe22"]
        assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 2)
        assert [ax.get_visible() for ax in axes] == [True, True, True, False]
        assert [len(ax.lines) > 0 for ax in axes] == [True, True, True, False]

    def test_single_dataset_on_larger_grid(self, sample_1d_dataset):
        axes = multiplot(sample_1d_dataset, nrow=2, ncol=2, show=False)

        assert axes.shape == (4,)
        assert [ax.get_visible() for ax in axes] == [True, False, False, False]

    def test_shared_x_labels_kept_above_hidden_panel(self, sample_1d_dataset):
        axes = multiplot(
            [sample_1d_dataset] * 3, nrow=2, ncol=2, sharex=True, show=False
        )

        # axe11 has axe21 below it; axe12 has only a hidden panel below it
        assert not axes[0].xaxis.label.get_visible()
        assert axes[1].xaxis.label.get_visible()
        assert axes[2].xaxis.label.get_visible()

    @pytest.mark.parametrize(
        "grid",
        [
            {"nrow": 1, "ncol": 2},
            {"nrows": 1, "ncols": 2},
            {"nrow": 1, "ncols": 2},
            {"nrow": 1, "ncol": 1},
        ],
    )
    def test_explicit_grid_too_small_raises(self, sample_1d_dataset, grid):
        with pytest.raises(ValueError, match="fewer panels than the 3 datasets"):
            multiplot([sample_1d_dataset] * 3, show=False, **grid)

    @pytest.mark.parametrize(
        ("count", "grid", "geometry"),
        [
            (4, {"nrow": 2}, (2, 2)),
            (4, {"ncol": 2}, (2, 2)),
            (3, {"nrow": 1}, (1, 3)),
            (3, {"ncol": 1}, (3, 1)),
            (3, {"nrows": 2}, (2, 2)),
            (3, {"ncols": 2}, (2, 2)),
            (5, {"ncol": 3}, (2, 3)),
            (2, {"nrow": 3}, (3, 1)),
        ],
    )
    def test_missing_grid_dimension_is_inferred(
        self, sample_1d_dataset, count, grid, geometry
    ):
        axes = multiplot([sample_1d_dataset] * count, show=False, **grid)

        assert axes[0].get_subplotspec().get_gridspec().get_geometry() == geometry
        assert [ax.get_visible() for ax in axes] == [True] * count + [False] * (
            geometry[0] * geometry[1] - count
        )

    @pytest.mark.parametrize(
        "grid",
        [
            {"nrow": 0},
            {"ncol": -1},
            {"nrows": 0},
            {"ncols": -1},
            {"nrow": 2, "ncol": 0},
            {"nrow": -1, "ncol": 2},
            {"nrow": 1.5},
        ],
    )
    def test_non_positive_grid_size_raises(self, sample_1d_dataset, grid):
        with pytest.raises(ValueError, match="must be a positive integer"):
            multiplot([sample_1d_dataset] * 2, show=False, **grid)

    @pytest.mark.parametrize(
        ("datasets", "options"),
        [([], {}), (None, {}), ([], {"sharex": True}), ([], {"nrow": 2})],
    )
    def test_no_dataset_raises(self, datasets, options):
        with pytest.raises(ValueError, match="No dataset to plot"):
            multiplot(datasets, show=False, **options)

    @pytest.mark.parametrize(
        ("count", "geometry"),
        [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2)), (5, (2, 3)), (6, (2, 3))],
    )
    def test_automatic_grid_without_nrow_ncol(self, sample_1d_dataset, count, geometry):
        axes = multiplot([sample_1d_dataset] * count, show=False)

        assert axes[0].get_subplotspec().get_gridspec().get_geometry() == geometry
        assert axes.shape == (geometry[0] * geometry[1],)
        assert [ax.get_visible() for ax in axes[:count]] == [True] * count
        assert not any(ax.get_visible() for ax in axes[count:])
        assert all(len(ax.lines) > 0 for ax in axes[:count])

    def test_with_transposed_unchanged(self, sample_2d_dataset):
        axes = multiplot(sample_2d_dataset, method="with_transposed", show=False)

        assert axes.shape == (2,)
        assert all(ax.get_visible() for ax in axes)
