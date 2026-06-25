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

    def test_interpolate_incompatible_coord_units_message(self):
        """Interpolation should identify the operation and incompatible units."""
        ds = NDDataset(
            np.array([0.0, 2.0, 4.0]),
            coordset=[
                Coord(np.array([4000.0, 3000.0, 2000.0]), title="x", units="cm^-1")
            ],
        )
        target = Coord(np.array([1.0, 2.0, 3.0]), title="x", units="s")

        with pytest.raises(ValueError) as exc:
            ds.interpolate(dim="x", coord=target)
        message = str(exc.value)
        assert "Cannot interpolate coordinates" in message
        assert "dimension 'x'" in message
        assert "s" in message
        assert "cm" in message
        assert "Convert the coordinates to compatible units before retrying." in message


class TestInterpolateDecreasingCoord:
    """Interpolation must honour the target order even for decreasing axes (#1100)."""

    @pytest.mark.parametrize("method", ["linear", "pchip"])
    def test_decreasing_source_increasing_target(self, method):
        # spectra are commonly stored as decreasing wavenumbers (4000 -> 400)
        x = np.linspace(4000.0, 400.0, 9)
        ds = NDDataset(x.copy(), coordset=[Coord(x, title="wn")])

        target = np.linspace(500.0, 3500.0, 7)  # increasing
        result = ds.interpolate(dim="x", coord=target, method=method)

        out = result.coord("x").data
        # the result follows the requested (increasing) order, not the source's
        assert np.all(np.diff(out) > 0)
        np.testing.assert_allclose(out, target, rtol=1e-9)
        np.testing.assert_allclose(result.data, target, rtol=1e-9)

    def test_decreasing_source_decreasing_target(self):
        x = np.linspace(4000.0, 400.0, 9)
        ds = NDDataset(x.copy(), coordset=[Coord(x, title="wn")])

        target = np.linspace(3500.0, 500.0, 7)  # decreasing
        result = ds.interpolate(dim="x", coord=target)

        out = result.coord("x").data
        assert np.all(np.diff(out) < 0)
        np.testing.assert_allclose(out, target, rtol=1e-9)
        np.testing.assert_allclose(result.data, target, rtol=1e-9)

    def test_decreasing_source_mask_follows_data(self):
        # When the source coordinate is decreasing, the mask must be reordered
        # together with the data, so a masked sample keeps its own coordinate
        # value after interpolation onto an increasing target (follow-up to #1100).
        x = np.linspace(5.0, 1.0, 5)  # decreasing 5 -> 1
        data = x.copy()  # data[i] == x[i]
        mask = np.array([False, True, False, False, False])  # mask the x == 4 sample
        ds = NDDataset(data, coordset=[Coord(x, title="x")], mask=mask)

        result = ds.interpolate(dim="x", coord=np.linspace(1.0, 5.0, 5))

        out_x = result.coord("x").data
        np.testing.assert_allclose(result.data, out_x)  # data identity preserved
        # the x == 4 node stays masked, not its mirror image x == 2
        np.testing.assert_allclose(out_x[np.asarray(result.mask)], [4.0])

    def test_decreasing_source_secondary_coord_follows_primary(self):
        # A secondary coordinate sharing a decreasing primary must be reordered
        # with the primary's sort order; otherwise it comes out reversed relative
        # to the interpolated primary (follow-up to #1100).
        from spectrochempy import CoordSet

        xp = np.array([5.0, 4.0, 2.5, 1.5, 1.0])  # decreasing primary
        xs = np.array([100.0, 64.0, 30.0, 18.0, 9.0])  # secondary, grows with xp
        multi = CoordSet(Coord(xp, title="wn"), Coord(xs, title="sec"), name="x")
        # make the decreasing primary the default (the interpolation axis)
        multi.select([c.title for c in multi.coords].index("wn") + 1)
        ds = NDDataset(np.arange(5.0), coordset=[multi])

        target = np.array([1.2, 2.0, 3.0, 4.0, 4.8])  # increasing
        result = ds.interpolate(dim="x", coord=target)

        rc = result.coord("x")
        out_sec = np.asarray(next(c for c in rc.coords if c.title == "sec").data)
        # the secondary tracks the now-increasing primary instead of reversing
        assert np.all(np.diff(out_sec) > 0)
        order = np.argsort(xp)
        np.testing.assert_allclose(
            out_sec, np.interp(target, xp[order], xs[order]), rtol=1e-3
        )


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


class TestInterpolateFillValue:
    """``fill_value`` handling, consistent between linear and PCHIP (#1093)."""

    @staticmethod
    def _dataset():
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x**2
        return NDDataset(y, coordset=[Coord(x, title="x")])

    # -1.0 and 5.0 fall outside the original [0, 4] range; 0.5 and 2.5 are inside.
    new_x = np.array([-1.0, 0.5, 2.5, 5.0])

    @pytest.mark.parametrize("method", ["linear", "pchip"])
    def test_default_fill_value_is_nan_outside_range(self, method):
        """Both methods leave out-of-range points as NaN by default."""
        result = self._dataset().interpolate(dim="x", coord=self.new_x, method=method)
        assert np.isnan(result.data[0])
        assert np.isnan(result.data[-1])
        assert not np.isnan(result.data[1])
        assert not np.isnan(result.data[2])

    @pytest.mark.parametrize("method", ["linear", "pchip"])
    def test_constant_fill_value_outside_range(self, method):
        """A finite ``fill_value`` fills out-of-range points for both methods."""
        result = self._dataset().interpolate(
            dim="x", coord=self.new_x, method=method, fill_value=0.0
        )
        assert result.data[0] == 0.0
        assert result.data[-1] == 0.0

    def test_pchip_fill_value_leaves_in_range_unchanged(self):
        """Changing ``fill_value`` only affects out-of-range points for PCHIP."""
        ds = self._dataset()
        ref = ds.interpolate(dim="x", coord=self.new_x, method="pchip")
        filled = ds.interpolate(
            dim="x", coord=self.new_x, method="pchip", fill_value=0.0
        )
        np.testing.assert_allclose(filled.data[1:3], ref.data[1:3], rtol=1e-10)
        assert filled.data[0] == 0.0
        assert filled.data[-1] == 0.0


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
    """Label handling tests (point-wise carry-over policy, GH #1098)."""

    def test_interpolate_resampled_points_unlabelled(self):
        """Genuinely resampled points (no exact match) are left unlabelled."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = ["a", "b", "c", "d", "e"]
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", labels=labels)])

        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        # none of the target points match an original value -> unlabelled
        assert coord._labels is None

    def test_interpolate_identity_preserves_all_labels(self):
        """An identity target grid keeps every label."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = ["a", "b", "c", "d", "e"]
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", labels=labels)])

        result = ds.interpolate(dim="x", coord=x.copy())

        coord = result.coord("x")
        assert coord.is_labeled
        assert list(np.asarray(coord.get_labels())) == labels

    def test_interpolate_reordered_preserves_labels_at_values(self):
        """A reordered target grid keeps labels attached to their values."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = ["a", "b", "c", "d", "e"]
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", labels=labels)])

        new_x = np.array([2.0, 0.0, 4.0, 1.0, 3.0])
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        assert list(np.asarray(coord.get_labels())) == ["c", "a", "e", "b", "d"]

    def test_interpolate_subset_preserves_matching_labels(self):
        """A subset target grid keeps only the matching labels."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = ["a", "b", "c", "d", "e"]
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", labels=labels)])

        new_x = np.array([1.0, 3.0])
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        assert list(np.asarray(coord.get_labels())) == ["b", "d"]

    def test_interpolate_mixed_grid_labels_exact_matches_only(self):
        """A mixed grid labels exact matches and leaves new points unlabelled."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = ["a", "b", "c", "d", "e"]
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", labels=labels)])

        new_x = np.array([0.0, 1.5, 2.0, 3.5])
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        assert list(np.asarray(coord.get_labels())) == ["a", "", "c", ""]

    def test_interpolate_no_match_gives_unlabelled_coord(self):
        """When no target point matches, the coordinate stays unlabelled."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = ["a", "b", "c", "d", "e"]
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ds = NDDataset(y, coordset=[Coord(x, title="x", labels=labels)])

        new_x = np.array([0.1, 1.1, 2.1])
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        assert not coord.is_labeled
        assert coord._labels is None

    def test_interpolate_secondary_coord_labels_consistent(self):
        """Same-dim secondary coordinates follow the same point-wise policy."""
        from spectrochempy import CoordSet

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 10.0, 20.0])
        data = np.arange(15).reshape(3, 5).astype(float)

        primary = Coord(x, title="x", labels=["a", "b", "c", "d", "e"])
        secondary = Coord(10.0 * x, title="x10", labels=["p", "q", "r", "s", "t"])
        multi_coord = CoordSet(primary, secondary, name="x")
        ds = NDDataset(data, coordset=[Coord(y, title="y"), multi_coord])

        # Matching is driven by the default coordinate; pick a target that is a
        # reordered subset of its own values (original positions 3 then 1).
        default_values = np.asarray(multi_coord.default.data)
        new_x = default_values[[3, 1]]
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        assert isinstance(coord, CoordSet)
        # every same-dim coordinate carries its own labels for the matched
        # points, consistently with the default/primary coordinate
        for cc in coord.coords:
            labels = list(np.asarray(cc.get_labels()))
            assert cc.is_labeled
            assert labels[0] != "" and labels[1] != ""


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


class TestInterpolateCoordReconstruction:
    """Targeted tests for coordinate reconstruction via _interpolate_dim."""

    def test_interpolate_simple_coord_replacement(self):
        """
        Simple coord replacement preserves target values and name, and leaves
        genuinely resampled points unlabelled.
        """
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        labels = ["a", "b", "c", "d", "e"]
        ds = NDDataset(y, coordset=[Coord(x, title="x", units="m", labels=labels)])

        new_x = np.array([0.5, 1.5, 2.5])
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        np.testing.assert_allclose(coord.data, [0.5, 1.5, 2.5])
        assert coord.name == "x"
        assert coord.labels is None

    def test_interpolate_multi_coord_reconstruction(self):
        """
        Multi-coord reconstruction preserves group nature and default, leaves
        resampled points unlabelled, and interpolates secondary numeric data.
        """
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 10.0, 20.0])
        data = np.arange(15).reshape(3, 5).astype(float)

        coord_x = Coord(x, title="x", labels=["z0", "z1", "z2", "z3", "z4"])
        secondary = Coord(x**2, title="x^2")
        coord_y = Coord(y, title="y")

        from spectrochempy import CoordSet

        multi_coord = CoordSet(coord_x, secondary, name="x")
        ds = NDDataset(data, coordset=[coord_y, multi_coord])
        assert isinstance(ds.coord("x"), CoordSet)

        new_x = np.array([0.5, 1.5, 2.5])
        result = ds.interpolate(dim="x", coord=new_x)

        coord = result.coord("x")
        assert isinstance(coord, CoordSet)
        assert coord.is_same_dim
        assert coord.default_index == multi_coord.default_index
        assert coord.name == "x"

        primary = coord.default
        assert primary is not None
        assert primary.labels is None
        np.testing.assert_allclose(primary.data, [0.5, 1.5, 2.5])

        # Verify secondary coord exists and was interpolated
        # -- resampled (non-matching) points leave children unlabelled, length 3
        assert len(coord) > 1


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

    def test_interpolate_array_target_preserves_coord_units_title(self):
        """A plain-array target keeps the source coordinate units/title (#1094)."""
        x = np.linspace(4000.0, 400.0, 10)
        ds = NDDataset(
            np.linspace(0.0, 1.0, 10),
            coordset=[Coord(x, title="wavenumber", units="cm^-1")],
        )

        new_x = np.linspace(3500.0, 500.0, 6)  # bare ndarray, no metadata
        result = ds.interpolate(dim="x", coord=new_x)

        assert result.coord("x").units == ds.coord("x").units
        assert result.coord("x").title == ds.coord("x").title
        # coordinate values follow the requested target, units only attached
        np.testing.assert_allclose(result.coord("x").data, new_x, rtol=1e-10)

    def test_interpolate_explicit_coord_target_keeps_own_metadata(self):
        """An explicit Coord target keeps its own title, not the source's (#1094)."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ds = NDDataset(
            np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
            coordset=[Coord(x, title="wavenumber", units="cm^-1")],
        )

        target = Coord(np.array([0.5, 1.5, 2.5, 3.5]), title="mine", units="cm^-1")
        result = ds.interpolate(dim="x", coord=target)

        assert result.coord("x").title == "mine"

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


class TestInterpolateDimResolution:
    """
    Regression tests for dim/dims argument resolution.

    The get_axis helper pops the ``dims`` key first; if ``dims=None``
    is explicitly in scope (the default), the ``dim`` keyword is never
    consulted, causing ``dim=0`` / ``dim="y"`` to silently fall back to
    the last axis.  These tests verify that every supported dim
    specification form is honoured correctly.
    """

    @pytest.fixture
    def ds(self):
        """2D dataset with dims ['y', 'x'] and shape (5, 7)."""
        x = Coord(np.linspace(4000.0, 1000.0, 7), title="wavenumber", units="cm^-1")
        y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
        return NDDataset(
            np.arange(35.0, dtype="float64").reshape(5, 7),
            coordset=[y, x],
        )

    def test_dim_0_interpolates_first_axis(self, ds):
        """dim=0 must interpolate axis 0 (y dimension)."""
        target = np.linspace(10.0, 50.0, 3)
        r = ds.interpolate(dim=0, coord=target)
        assert r.shape == (3, 7)
        np.testing.assert_allclose(r.y.data, target, rtol=1e-10)

    def test_dim_y_interpolates_y_dimension(self, ds):
        """dim='y' must interpolate the y dimension."""
        target = np.linspace(10.0, 50.0, 3)
        r = ds.interpolate(dim="y", coord=target)
        assert r.shape == (3, 7)
        np.testing.assert_allclose(r.y.data, target, rtol=1e-10)

    def test_dim_1_interpolates_last_axis(self, ds):
        """dim=1 must interpolate axis 1 (x dimension)."""
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds.interpolate(dim=1, coord=target)
        assert r.shape == (5, 3)
        np.testing.assert_allclose(r.x.data, target, rtol=1e-10)

    def test_dim_negative_one_interpolates_last_axis(self, ds):
        """dim=-1 must interpolate the last axis (x)."""
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds.interpolate(dim=-1, coord=target)
        assert r.shape == (5, 3)
        np.testing.assert_allclose(r.x.data, target, rtol=1e-10)

    def test_dim_x_interpolates_x_dimension(self, ds):
        """dim='x' must interpolate the x dimension."""
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds.interpolate(dim="x", coord=target)
        assert r.shape == (5, 3)
        np.testing.assert_allclose(r.x.data, target, rtol=1e-10)

    def test_default_dim_none_interpolates_last(self, ds):
        """Default dim=None must interpolate the last dimension (x)."""
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds.interpolate(coord=target)
        assert r.shape == (5, 3)
        np.testing.assert_allclose(r.x.data, target, rtol=1e-10)

    def test_dims_keyword_passed_explicitly_none(self, ds):
        """Explicit dims=None must not shadow dim for the default path."""
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds.interpolate(dims=None, coord=target)
        assert r.shape == (5, 3)
        np.testing.assert_allclose(r.x.data, target, rtol=1e-10)

    def test_1d_default_interpolates_last(self):
        """1D dataset: default dim must interpolate the only dimension."""
        x = Coord(np.linspace(0.0, 10.0, 5), title="x")
        ds = NDDataset(np.arange(5.0), coordset=[x])
        target = np.array([2.0, 4.0, 6.0])
        r = ds.interpolate(coord=target)
        assert r.shape == (3,)
        np.testing.assert_allclose(r.x.data, target, rtol=1e-10)
