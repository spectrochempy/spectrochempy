# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Units and labeling stateless behavior tests.

Tests automatic axis labeling from coordinates,
unit formatting, and unitless coordinate handling.
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


class TestUnitsLabeling:
    """Test units and axis labeling in stateless architecture."""

    def test_automatic_axis_labels(self, sample_1d_dataset):
        """Test 16: Automatic axis labels from coordinates."""
        ds_before = sample_1d_dataset.__dict__.copy()

        # Plot without custom labels
        ax = sample_1d_dataset.plot()

        # Verify automatic labeling uses coordinate titles and units
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        # Should contain coordinate information
        assert "Wavenumber" in xlabel, "X-axis should use coordinate title"
        assert "cm" in xlabel and "-1" in xlabel, "X-axis should use coordinate units"
        assert "Intensity" in ylabel, "Y-axis should use data title"
        assert "a" in ylabel and "u" in ylabel, "Y-axis should use data units"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_complex_unit_formatting(self):
        """Test 17: Complex unit formatting (superscripts, Greek letters)."""
        import numpy as np

        from spectrochempy import Coord, NDDataset

        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = np.random.random(20)

        # Dataset with complex units
        x_coord = Coord(data=x, title="Energy", units="kJ/mol")
        dataset = NDDataset(y, title="Test Signal", coordset=[x_coord])

        ds_before = dataset.__dict__.copy()

        # Plot and check formatting
        ax = dataset.plot()
        xlabel = ax.get_xlabel()

        # Should contain the units properly formatted
        assert "Energy" in xlabel, "Should contain coordinate title"
        assert "kJ" in xlabel or "mol" in xlabel, "Should contain units"

        # Test with special characters
        x_coord2 = Coord(data=x, title="Length", units="µs")
        dataset2 = NDDataset(y, title="Test Signal", coordset=[x_coord2])

        ax2 = dataset2.plot()
        xlabel2 = ax2.get_xlabel()

        assert "Length" in xlabel2, "Should contain coordinate title"

        # Verify datasets unchanged
        assert_dataset_state_unchanged(ds_before, dataset)

    def test_unitless_coordinates(self):
        """Test 18: Unitless coordinate handling."""
        import numpy as np

        from spectrochempy import Coord, NDDataset

        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = np.random.random(20)

        # Dataset without units
        x_coord = Coord(data=x, title="Position")
        dataset = NDDataset(y, title="Test Signal", coordset=[x_coord])

        ds_before = dataset.__dict__.copy()

        # Plot and check clean labeling
        ax = dataset.plot()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        # Should not have unit suffixes
        assert "Position" in xlabel, "Should contain coordinate title"
        # Should not have things like "/dimensionless" or similar
        assert "/dimensionless" not in xlabel, (
            "Should not contain unit suffix for unitless"
        )

        # Y-axis should have data title without unit suffix
        assert "Test Signal" in ylabel, "Should contain data title"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, dataset)
