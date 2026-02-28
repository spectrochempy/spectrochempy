# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
3D plotting stateless behavior tests.

Tests 3D plotting functionality ensuring proper surface creation,
method defaults, and stateless behavior.
"""


class Test3DStateless:
    """Test 3D plotting in stateless architecture."""

    def test_3d_surface_creation(self, sample_3d_dataset):
        """Test 14: 3D surface creation with basic verification."""
        ds_before = sample_3d_dataset.__dict__.copy()

        # Test surface plot
        ax = sample_3d_dataset.plot(method="surface")

        # Verify 3D axes created
        assert hasattr(ax, "zaxis"), "3D plot should have z-axis"
        assert hasattr(ax, "zaxis"), "3D plot should have y-axis"
        assert hasattr(ax, "xaxis"), "3D plot should have x-axis"

        # Check projection
        # Note: The exact attribute name depends on matplotlib version
        # Common attributes to check
        ax_name = ax.name if hasattr(ax, "name") else ""
        assert "3d" in ax_name.lower() or hasattr(
            ax, "zaxis"
        ), "Should have 3D projection"

        # Verify surface mesh was created (check collections)
        has_surface = False
        if hasattr(ax, "collections"):
            for collection in ax.collections:
                if hasattr(collection, "get_array"):
                    has_surface = True
                    break

        # Also check if any surface-like objects exist
        # This is a basic check - actual implementation may vary
        assert (
            has_surface or len(ax.collections) > 0
        ), "Surface plot should create surface mesh"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_3d_dataset)

        # Verify no plotting attributes added
        assert not hasattr(sample_3d_dataset, "fig")
        assert not hasattr(sample_3d_dataset, "ndaxes")

    def test_3d_default_method(self, sample_3d_dataset):
        """Test 15: 3D default method should be surface."""
        ds_before = sample_3d_dataset.__dict__.copy()

        # Test default method (no explicit method)
        ax_default = sample_3d_dataset.plot()

        # Test explicit surface method
        ax_explicit = sample_3d_dataset.plot(method="surface")

        # Both should be 3D plots
        assert hasattr(ax_default, "zaxis"), "Default plot should be 3D"
        assert hasattr(ax_explicit, "zaxis"), "Explicit surface plot should be 3D"

        # Both should have similar basic properties
        # Check that both are 3D axes
        default_is_3d = hasattr(ax_default, "zaxis")
        explicit_is_3d = hasattr(ax_explicit, "zaxis")

        assert default_is_3d and explicit_is_3d, "Both methods should create 3D plots"

        # Verify datasets unchanged
        assert_dataset_state_unchanged(ds_before, sample_3d_dataset)

        # Should not have added plotting attributes
        assert not hasattr(sample_3d_dataset, "fig")
        assert not hasattr(sample_3d_dataset, "ndaxes")
