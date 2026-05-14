# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the NDDataset.interpolate method."""

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset


class TestInterpolateBasic:
    """Basic interpolation tests."""

    def test_interpolate_linear_basic(self):
        """Test basic linear interpolation on 1D data."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        assert result.shape == (4,)
        np.testing.assert_allclose(result.data, [1.0, 3.0, 5.0, 7.0], rtol=1e-5)

    def test_interpolate_pchip_basic(self):
        """Test PCHIP interpolation on 1D data."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x, method="pchip")

        assert result.shape == (4,)

    def test_interpolate_coord_input(self):
        """Test interpolation with Coord as target."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_coord = Coord(np.array([0.5, 1.5, 2.5, 3.5]), title="x")
        result = ds.interpolate(dim="x", coord=new_coord)

        assert result.shape == (4,)
        np.testing.assert_allclose(result.data, [1.0, 3.0, 5.0, 7.0], rtol=1e-5)

    def test_interpolate_nddataset_input(self):
        """Test interpolation with NDDataset as target coordinate."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        target_ds = NDDataset(
            np.array([0.5, 1.5, 2.5, 3.5]),
            coordset=[Coord(np.array([0.5, 1.5, 2.5, 3.5]), title="x")],
        )
        result = ds.interpolate(dim="x", coord=target_ds)

        assert result.shape == (4,)

    def test_interpolate_unsorted_coordinates(self):
        """Test interpolation handles unsorted coordinates."""
        x = np.array([4.0, 2.0, 0.0, 1.0, 3.0])
        y = np.array([8.0, 4.0, 0.0, 2.0, 6.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        assert result.shape == (4,)
        np.testing.assert_allclose(result.data, [1.0, 3.0, 5.0, 7.0], rtol=1e-5)

    def test_interpolate_assume_sorted(self):
        """Test assume_sorted parameter."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x, assume_sorted=True)

        assert result.shape == (4,)


class TestInterpolateMultidimensional:
    """Multi-dimensional interpolation tests."""

    def test_interpolate_2d(self):
        """Test interpolation on 2D data along first dimension."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 10.0, 20.0])
        data = np.arange(15).reshape(3, 5).astype(float)
        ds = NDDataset(data, coordset=[Coord(y, title="y"), Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        assert result.shape == (3, 4)
        np.testing.assert_allclose(result.data[0], [0.5, 1.5, 2.5, 3.5], rtol=1e-5)

    def test_interpolate_multiple_dims(self):
        """Test interpolation along multiple dimensions."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 10.0, 20.0])
        data = np.arange(12).reshape(3, 4).astype(float)
        ds = NDDataset(data, coordset=[Coord(y, title="y"), Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5])
        new_y = np.array([5.0, 15.0])
        result = ds.interpolate(dims=["x", "y"], coord={"x": new_x, "y": new_y})

        assert result.shape == (2, 3)


class TestInterpolateErrorCases:
    """Error handling tests."""

    def test_duplicate_coordinates_error(self):
        """Test that duplicate coordinates raise error."""
        x = np.array([0.0, 1.0, 1.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        with pytest.raises(ValueError, match="Duplicate"):
            ds.interpolate(dim="x", coord=new_x, assume_sorted=False)

    def test_non_numeric_coordinate_error(self):
        """Test that non-numeric coordinates raise TypeError."""
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y)
        ds.set_coordset(x=None)

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        with pytest.raises(ValueError):
            ds.interpolate(dim="x", coord=new_x)


class TestInterpolateUnits:
    """Unit handling tests."""

    def test_interpolate_with_units(self):
        """Test interpolation with unit conversion."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", units="m")])

        new_coord = Coord(np.array([0.5, 1.5, 2.5, 3.5]), title="x", units="m")
        result = ds.interpolate(dim="x", coord=new_coord)

        assert result.shape == (4,)
        assert str(result.coord("x").units) == "m"

    def test_interpolate_unit_conversion(self):
        """Test interpolation with automatic unit conversion."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", units="m")])

        new_coord = Coord(np.array([50.0, 150.0, 250.0, 350.0]), title="x", units="cm")
        result = ds.interpolate(dim="x", coord=new_coord)

        assert result.shape == (4,)


class TestInterpolateLabels:
    """Label handling tests."""

    def test_interpolate_removes_labels(self):
        """Test that labels are removed after interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = ["a", "b", "c", "d", "e"]
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", labels=labels)])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        assert coord._labels is None


class TestInterpolateMask:
    """Mask handling tests."""

    def test_interpolate_masked_data(self):
        """Test interpolation of masked data."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        mask = np.array([False, False, False, False, False])
        ds = NDDataset(y, coordset=[Coord(x, title="x")], mask=mask)

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        assert result.shape == (4,)
        assert not result.is_masked


class TestInterpolateMultipleCoords:
    """Multiple coordinates per dimension tests."""

    def test_interpolate_multiple_coords(self):
        """Test interpolation with multiple coordinates per dimension."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 10.0, 20.0])
        data = np.arange(15).reshape(3, 5).astype(float)

        coord_x = Coord(x, title="x")
        coord_y = Coord(y, title="y")

        ds = NDDataset(data, coordset=[coord_y, coord_x])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        assert result.shape == (3, 4)


class TestInterpolateInplace:
    """Inplace modification tests."""

    def test_interpolate_inplace(self):
        """Test inplace interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x, inplace=True)

        assert result is ds
        assert result.shape == (4,)


class TestInterpolateMetadata:
    """Metadata preservation tests."""

    def test_interpolate_preserves_name_title(self):
        """Test that name and title are preserved."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(
            y, coordset=[Coord(x, title="x")], title="test title", name="test_name"
        )

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        assert result.name == "test_name"
        assert result.title == "test title"

    def test_interpolate_updates_history(self):
        """Test that history is updated."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        assert len(result.history) > 0


class TestInterpolateTolerance:
    """Monotonicity tolerance tests."""

    def test_interpolate_noisy_increasing(self):
        """Test that slightly noisy increasing coordinates pass."""
        x = np.array([0.0, 1.0 + 1e-13, 2.0 - 1e-13, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        assert result.shape == (4,)
        np.testing.assert_allclose(result.data, [1.0, 3.0, 5.0, 7.0], rtol=1e-5)

    def test_interpolate_near_duplicate_fails(self):
        """Test that near-duplicate coordinates (within tolerance) fail."""
        x = np.array([0.0, 1.0, 1.0 + 1e-13, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5])
        with pytest.raises(ValueError, match="Duplicate or near-duplicate"):
            ds.interpolate(dim="x", coord=new_x)

    def test_interpolate_true_duplicate_fails(self):
        """Test that true duplicate coordinates fail."""
        x = np.array([0.0, 1.0, 1.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x")])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        with pytest.raises(ValueError, match="Duplicate"):
            ds.interpolate(dim="x", coord=new_x, assume_sorted=False)


class TestAlignInterpolate:
    """Tests for align with interpolate method."""

    def test_align_interpolate_basic(self):
        """Test basic align with interpolate method."""
        coord0 = Coord(
            data=np.linspace(4000.0, 1000.0, 10), units="cm^-1", title="wavenumber"
        )
        coord1 = Coord(data=np.linspace(0.0, 60.0, 100), units="s", title="time")

        ds1 = NDDataset(
            np.random.rand(100, 10),
            coordset=[coord1, coord0],
            title="test",
            units="absorbance",
        )

        coord0_2 = Coord(
            data=np.linspace(4000.0, 1000.0, 9), units="cm^-1", title="wavenumber"
        )
        coord1_2 = Coord(data=np.linspace(0.0, 60.0, 50), units="s", title="time")

        ds2 = NDDataset(
            np.random.rand(50, 9),
            coordset=[coord1_2, coord0_2],
            title="test",
            units="absorbance",
        )

        ds1_aligned, ds2_aligned = ds1.align(ds2, dim="x", method="interpolate")

        assert ds1_aligned.shape[1] == 10
        assert ds2_aligned.shape[1] == 10
        assert len(ds2_aligned.coord("x").data) == 10

    def test_align_interpolate_pchip(self):
        """Test align with interpolate and pchip method."""
        coord0 = Coord(
            data=np.linspace(4000.0, 1000.0, 10), units="cm^-1", title="wavenumber"
        )
        coord1 = Coord(data=np.linspace(0.0, 60.0, 100), units="s", title="time")

        ds1 = NDDataset(
            np.random.rand(100, 10),
            coordset=[coord1, coord0],
            title="test",
            units="absorbance",
        )

        coord0_2 = Coord(
            data=np.linspace(4000.0, 1000.0, 9), units="cm^-1", title="wavenumber"
        )
        coord1_2 = Coord(data=np.linspace(0.0, 60.0, 50), units="s", title="time")

        ds2 = NDDataset(
            np.random.rand(50, 9),
            coordset=[coord1_2, coord0_2],
            title="test",
            units="absorbance",
        )

        ds1_aligned, ds2_aligned = ds1.align(
            ds2, dim="x", method="interpolate", interpolate_method="pchip"
        )

        assert ds1_aligned.shape[1] == 10
        assert ds2_aligned.shape[1] == 10

    def test_align_interpolate_multiple_datasets(self):
        """Test align with interpolate method for 3 datasets."""
        coord0 = Coord(
            data=np.linspace(4000.0, 1000.0, 10), units="cm^-1", title="wavenumber"
        )
        coord1 = Coord(data=np.linspace(0.0, 60.0, 100), units="s", title="time")

        ds1 = NDDataset(
            np.random.rand(100, 10),
            coordset=[coord1, coord0],
            title="test",
            units="absorbance",
        )

        coord0_2 = Coord(
            data=np.linspace(4000.0, 1000.0, 9), units="cm^-1", title="wavenumber"
        )
        coord1_2 = Coord(data=np.linspace(0.0, 60.0, 50), units="s", title="time")
        ds2 = NDDataset(
            np.random.rand(50, 9),
            coordset=[coord1_2, coord0_2],
            title="test",
            units="absorbance",
        )

        coord0_3 = Coord(
            data=np.linspace(4000.0, 1000.0, 8), units="cm^-1", title="wavenumber"
        )
        coord1_3 = Coord(data=np.linspace(0.0, 60.0, 40), units="s", title="time")
        ds3 = NDDataset(
            np.random.rand(40, 8),
            coordset=[coord1_3, coord0_3],
            title="test",
            units="absorbance",
        )

        ds1_aligned, ds2_aligned, ds3_aligned = ds1.align(
            ds2, ds3, dim="x", method="interpolate"
        )

        assert ds1_aligned.shape[1] == 10
        assert ds2_aligned.shape[1] == 10
        assert ds3_aligned.shape[1] == 10

        np.testing.assert_array_equal(
            ds1_aligned.coord("x").data, ds2_aligned.coord("x").data
        )
        np.testing.assert_array_equal(
            ds1_aligned.coord("x").data, ds3_aligned.coord("x").data
        )
        assert len(ds2_aligned.coord("x").data) == 10
