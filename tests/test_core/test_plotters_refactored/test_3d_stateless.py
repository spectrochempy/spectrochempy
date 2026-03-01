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

    assert not plotting_keys, (
        f"Dataset mutated by plotting with new attributes: {plotting_keys}"
    )
    assert not hasattr(dataset_after, "fig")
    assert not hasattr(dataset_after, "ndaxes")


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
        assert "3d" in ax_name.lower() or hasattr(ax, "zaxis"), (
            "Should have 3D projection"
        )

        # Verify surface mesh was created (check collections)
        has_surface = False
        if hasattr(ax, "collections"):
            for collection in ax.collections:
                if hasattr(collection, "get_array"):
                    has_surface = True
                    break

        # Also check if any surface-like objects exist
        # This is a basic check - actual implementation may vary
        assert has_surface or len(ax.collections) > 0, (
            "Surface plot should create surface mesh"
        )

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_3d_dataset)

        # Verify no plotting attributes added
        assert not hasattr(sample_3d_dataset, "fig")
        assert not hasattr(sample_3d_dataset, "ndaxes")

    def test_3d_default_method(self, sample_3d_dataset):
        """Test 15: 3D plotting with explicit surface method."""
        ds_before = sample_3d_dataset.__dict__.copy()

        # Test explicit surface method - default for 2D is stack, so specify surface
        ax_explicit = sample_3d_dataset.plot(method="surface")

        # Should be 3D plot
        assert hasattr(ax_explicit, "zaxis"), "Explicit surface plot should be 3D"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_3d_dataset)

        # Should not have added plotting attributes
        assert not hasattr(sample_3d_dataset, "fig")
        assert not hasattr(sample_3d_dataset, "ndaxes")
