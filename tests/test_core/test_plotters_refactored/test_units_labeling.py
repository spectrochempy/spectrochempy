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
        assert "cm^-1" in xlabel, "X-axis should use coordinate units"
        assert "Intensity" in ylabel, "Y-axis should use data title"
        assert "a.u." in ylabel, "Y-axis should use data units"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, sample_1d_dataset)

    def test_complex_unit_formatting(self):
        """Test 17: Complex unit formatting (superscripts, Greek letters)."""
        # Create dataset with complex units
        import numpy as np

        from spectrochempy import NDDataset

        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = np.random.random(20)

        # Dataset with complex units
        dataset = NDDataset(y, title="Test Signal")
        dataset.set_coordset(x, title="Energy", units="kJ/mol")

        ds_before = dataset.__dict__.copy()

        # Plot and check formatting
        ax = dataset.plot()
        xlabel = ax.get_xlabel()

        # Should contain the units properly formatted
        assert "Energy" in xlabel, "Should contain coordinate title"
        assert "kJ/mol" in xlabel, "Should contain units"

        # Test with special characters
        dataset2 = NDDataset(y, title="Test Signal")
        dataset2.set_coordset(x, title="Length", units="µs")

        ax2 = dataset2.plot()
        xlabel2 = ax2.get_xlabel()

        assert "Length" in xlabel2, "Should contain coordinate title"
        assert "µs" in xlabel2, "Should contain microsecond symbol"

        # Verify datasets unchanged
        assert_dataset_state_unchanged(ds_before, dataset)

    def test_unitless_coordinates(self):
        """Test 18: Unitless coordinate handling."""
        import numpy as np

        from spectrochempy import NDDataset

        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = np.random.random(20)

        # Dataset without units
        dataset = NDDataset(y, title="Test Signal")
        dataset.set_coordset(x, title="Position")  # No units specified

        ds_before = dataset.__dict__.copy()

        # Plot and check clean labeling
        ax = dataset.plot()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        # Should not have unit suffixes
        assert "Position" in xlabel, "Should contain coordinate title"
        # Should not have things like "/dimensionless" or similar
        assert (
            "/dimensionless" not in xlabel
        ), "Should not contain unit suffix for unitless"
        assert (
            "/" not in xlabel.split()[-1] if xlabel else False
        ), "Should not have trailing slash for unitless"

        # Y-axis should have data title without unit suffix
        assert "Test Signal" in ylabel, "Should contain data title"

        # Verify dataset unchanged
        assert_dataset_state_unchanged(ds_before, dataset)
