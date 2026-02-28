# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Matplotlib integration stateless behavior tests.

Tests matplotlib integration, state isolation, and resource management
in stateless architecture.
"""


def assert_dataset_state_unchanged(dataset_before, dataset_after):
    """Verify dataset was not mutated by plotting."""
    before_dict = (
        dataset_before if isinstance(dataset_before, dict) else dataset_before.__dict__
    )
    assert before_dict == dataset_after.__dict__, "Dataset mutated by plotting"
    assert not hasattr(dataset_after, "fig")
    assert not hasattr(dataset_after, "ndaxes")


class TestMatplotlibIntegration:
    """Test matplotlib integration in stateless architecture."""

    def test_stateless_resource_management(self, sample_1d_dataset):
        """Test 19: Stateless resource management and figure lifecycle."""
        ds_before = sample_1d_dataset.__dict__.copy()

        # Plot with SCP style
        ax1 = sample_1d_dataset.plot(style="paper")
        fig1 = ax1.figure

        # Verify figure is created and usable
        assert fig1 is not None, "Figure should be created"
        assert len(fig1.axes) > 0, "Figure should have axes"

        # Manual cleanup (user responsibility in stateless architecture)
        fig1.clf()  # Clear figure

        # Plot again on same dataset (should work fine)
        ax2 = sample_1d_dataset.plot(style="grayscale")
        fig2 = ax2.figure

        # Second plot should create fresh figure
        assert fig2 is not None, "Second plot should create new figure"
        assert len(fig2.axes) > 0, "Second figure should have axes"

        # Verify different styling applied
        # Both should be valid plots but potentially different appearance
        assert len(ax1.get_lines()) > 0, "First plot should be valid"
        assert len(ax2.get_lines()) > 0, "Second plot should be valid"

        # Manual cleanup of second figure
        fig2.clf()

        # Verify dataset unchanged throughout
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

        # Verify no plotting attributes added
        assert not hasattr(sample_1d_dataset, "fig")
        assert not hasattr(sample_1d_dataset, "ndaxes")

        # Verify cleanup worked
        assert len(fig1.axes) == 0, "First figure should be cleared"
        assert len(fig2.axes) == 0, "Second figure should be cleared"
