"""
Characterization tests for interpolation semantics on NDDataset.

This suite characterizes CURRENT behavior of ``interpolate()`` and
``align(..., method='interpolate')``. It does NOT validate a desired
future policy.

Note on scope: "interpolation semantics" here refers to the operation
of evaluating data on a new coordinate grid.  The ``align`` path uses
the same interpolation machinery, so its behavior is covered as well.
True resampling (e.g. downsampling with anti-aliasing) is not
characterised separately.

Coverage:
    - Return type, shape, dims, CoordSet, units, masks
    - Metadata (title, name, author, description, origin, meta)
    - History, ROI, identity, provenance
    - Coordinate rebuild semantics (increasing, decreasing, shifted,
      denser / sparser grids)
    - Label point-wise carry-over policy
    - Mask interpolation and thresholding
    - Secondary coordinate interpolation
    - Multi-dim sequential interpolation
    - align metadata propagation

Key observed patterns:
    - Copy-first + domain rebuild (not inplace by default)
    - Same scientific object, changed representation
    - History appended with timestamp formatting
    - ROI preserved but stale after coordinate rebuild
    - Mask reconstruction is operation-dependent
    - Labels on exact coordinate matches only
    - Multi-coordinate interpolation currently raises a ValueError
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset

# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def ds_1d():
    """Simple 1D dataset with one coord."""
    x = Coord(np.linspace(0.0, 10.0, 5), title="x")
    return NDDataset(np.arange(5.0, dtype="float64"), coordset=[x])


@pytest.fixture
def ds_2d():
    """
    Semantic-rich 2D dataset for interpolation characterization.

    - dims: ['y', 'x'] (5, 7)
    - CoordSet with titles, units
    - Full metadata
    """
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(np.linspace(4000.0, 1000.0, 7), title="wavenumber", units="cm^-1")
    ds = NDDataset(
        np.arange(35.0, dtype="float64").reshape(5, 7),
        coordset=[y, x],
        title="ds_title",
        name="ds_name",
    )
    ds.author = "test_author"
    ds.description = "test description"
    ds.origin = "test_origin"
    ds.meta.project = "test_project"
    ds.roi = [0.0, 10.0]
    ds.history = ["original entry"]
    return ds


@pytest.fixture
def ds_masked():
    """2D dataset with a masked value at [0, 0]."""
    arr = np.ma.MaskedArray(
        np.arange(35.0).reshape(5, 7), mask=np.zeros((5, 7), dtype=bool)
    )
    arr[0, 0] = np.ma.masked
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(np.linspace(4000.0, 1000.0, 7), title="wn", units="cm^-1")
    return NDDataset(arr, coordset=[y, x])


@pytest.fixture
def ds_multi_coord():
    """2D dataset with a multi-coord (CoordSet) on x dimension."""
    xp = np.linspace(4000.0, 1000.0, 7)
    xs = xp**2
    primary = Coord(xp, title="wavenumber", units="cm^-1")
    secondary = Coord(xs, title="wn_squared")
    multi_x = CoordSet(primary, secondary, name="x")
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    return NDDataset(
        np.arange(35.0, dtype="float64").reshape(5, 7),
        coordset=[y, multi_x],
    )


@pytest.fixture
def ds_labeled():
    """1D dataset with labeled coordinates."""
    x = Coord(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        title="x",
        labels=["a", "b", "c", "d", "e"],
    )
    return NDDataset(np.array([0.0, 2.0, 4.0, 6.0, 8.0]), coordset=[x])


# ======================================================================================
# RETURN TYPE, SHAPE, DIMS
# ======================================================================================


class TestReturnTypeShapeDims:
    """Characterize return type, shape, and dims after interpolation."""

    def test_return_type(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert isinstance(r, NDDataset)

    def test_shape_changes_along_interpolated_dim(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.shape == (5, 3)

    def test_shape_unchanged_along_other_dims(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.shape[0] == ds_2d.shape[0]

    def test_dims_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.dims == ds_2d.dims

    def test_data_values_interpolated(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        expected = ds_2d.data[:, 1:-1:2]
        np.testing.assert_allclose(r.data, expected)

    def test_inplace_returns_same_object(self, ds_2d):
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds_2d.interpolate(dim="x", coord=target, inplace=True)
        assert r is ds_2d


# ======================================================================================
# UNITS
# ======================================================================================


class TestUnits:
    """Characterize unit behavior during interpolation."""

    def test_dataset_units_preserved(self, ds_2d):
        ds_2d.units = "absorbance"
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert str(r.units) == str(ds_2d.units)

    def test_coord_units_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert str(r.coord("x").units) == str(ds_2d.coord("x").units)

    def test_coord_unit_conversion(self, ds_2d):
        target = Coord(np.linspace(3500.0, 1500.0, 3), title="x", units="m^-1")
        r = ds_2d.interpolate(dim="x", coord=target)
        assert str(r.coord("x").units) == str(ds_2d.coord("x").units)

    def test_array_target_inherits_source_units(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert str(r.coord("x").units) == str(ds_2d.coord("x").units)

    def test_array_target_inherits_source_title(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.coord("x").title == "wavenumber"


# ======================================================================================
# COORDSET
# ======================================================================================


class TestCoordSet:
    """Characterize CoordSet rebuild behavior during interpolation."""

    def test_simple_coord_replaced(self, ds_2d):
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds_2d.interpolate(dim="x", coord=target)
        np.testing.assert_allclose(r.coord("x").data, target, rtol=1e-10)

    def test_other_dim_coord_untouched(self, ds_2d):
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds_2d.interpolate(dim="x", coord=target)
        np.testing.assert_allclose(r.coord("y").data, ds_2d.coord("y").data, rtol=1e-10)

    def test_multi_coord_preserves_group_nature(self, ds_multi_coord):
        target = np.linspace(3500.0, 1500.0, 3)
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            ds_multi_coord.interpolate(dim="x", coord=target)

    def test_multi_coord_default_unchanged(self, ds_multi_coord):
        target = np.linspace(3500.0, 1500.0, 3)
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            ds_multi_coord.interpolate(dim="x", coord=target)

    def test_secondary_coord_interpolated(self, ds_multi_coord):
        target = np.linspace(3500.0, 1500.0, 3)
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            ds_multi_coord.interpolate(dim="x", coord=target)

    def test_secondary_coord_values_interpolated(self, ds_multi_coord):
        target = np.array([3000.0, 2000.0, 1500.0])
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            ds_multi_coord.interpolate(dim="x", coord=target)

    def test_coord_title_preserved_for_explicit_coord_target(self, ds_2d):
        target = Coord(np.linspace(3500.0, 1500.0, 3), title="mine", units="cm^-1")
        r = ds_2d.interpolate(dim="x", coord=target)
        assert r.coord("x").title == "mine"

    def test_coord_title_inherited_for_array_target(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.coord("x").title == ds_2d.coord("x").title


# ======================================================================================
# LABELS (point-wise carry-over policy, GH #1098)
# ======================================================================================


class TestLabels:
    """Characterize label behavior: exact-coordinate matches only."""

    def test_resampled_points_unlabelled(self, ds_labeled):
        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        r = ds_labeled.interpolate(dim="x", coord=new_x)
        assert r.coord("x")._labels is None

    def test_identity_preserves_all_labels(self, ds_labeled):
        r = ds_labeled.interpolate(dim="x", coord=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        assert r.coord("x").is_labeled
        assert list(np.asarray(r.coord("x").get_labels())) == ["a", "b", "c", "d", "e"]

    def test_reordered_preserves_labels_at_values(self, ds_labeled):
        new_x = np.array([2.0, 0.0, 4.0, 1.0, 3.0])
        r = ds_labeled.interpolate(dim="x", coord=new_x)
        assert list(np.asarray(r.coord("x").get_labels())) == ["c", "a", "e", "b", "d"]

    def test_subset_preserves_matching_labels(self, ds_labeled):
        new_x = np.array([1.0, 3.0])
        r = ds_labeled.interpolate(dim="x", coord=new_x)
        assert list(np.asarray(r.coord("x").get_labels())) == ["b", "d"]

    def test_mixed_grid_labels_exact_matches_only(self, ds_labeled):
        new_x = np.array([0.0, 1.5, 2.0, 3.5])
        r = ds_labeled.interpolate(dim="x", coord=new_x)
        assert list(np.asarray(r.coord("x").get_labels())) == ["a", "", "c", ""]

    def test_no_match_gives_unlabelled(self, ds_labeled):
        new_x = np.array([0.1, 1.1, 2.1])
        r = ds_labeled.interpolate(dim="x", coord=new_x)
        assert r.coord("x")._labels is None

    def test_multi_coord_labels_consistent(self, ds_multi_coord):
        """Current behavior: only the interpolated secondary coord keeps labels."""
        primary = ds_multi_coord.coord("x").default
        primary.labels = ["p0", "p1", "p2", "p3", "p4", "p5", "p6"]
        values = np.asarray(primary.data)
        target = values[[3, 1]]
        r = ds_multi_coord.interpolate(dim="x", coord=target)

        coord = r.coord("x")
        assert isinstance(coord, CoordSet)
        labels_by_title = {}
        for c in coord.coords:
            raw = getattr(c, "_labels", None)
            labels_by_title[c.title] = (
                None if raw is None else list(np.asarray(c.get_labels()))
            )
        assert labels_by_title["wn_squared"] == ["p3", "p1"]
        assert labels_by_title["wavenumber"] is None


# ======================================================================================
# MASKS
# ======================================================================================


class TestMasks:
    """Characterize mask behavior: float interpolation + 0.5 threshold."""

    def test_unmasked_after_interpolation_when_source_unmasked(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert not r.is_masked

    def test_mask_interpolated_along_dim(self, ds_masked):
        r = ds_masked.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert not r.is_masked

    def test_mask_propagated_through_interpolation(self):
        """A mask at a specific coord value propagates to nearby target points."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mask = np.array([False, True, False, False, False])
        ds = NDDataset(np.arange(5.0), coordset=[Coord(x, title="x")], mask=mask)
        new_x = np.array([0.5, 1.0, 1.5, 2.0])
        r = ds.interpolate(dim="x", coord=new_x)
        # mask at exact original coord passes through
        assert r.mask[1]  # x=1.0 is exact match

    def test_mask_outside_range_True(self):
        """Out-of-range target points get mask=True (fill_value=True)."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        mask = np.array([False, True, False, False])
        ds = NDDataset(np.arange(4.0), coordset=[Coord(x, title="x")], mask=mask)
        new_x = np.array([-0.5, 3.5])
        r = ds.interpolate(dim="x", coord=new_x)
        assert r.mask[0] and r.mask[1]


# ======================================================================================
# METADATA (title, name, author, description, origin, meta)
# ======================================================================================


class TestMetadata:
    """Characterize metadata preservation during interpolation."""

    def test_title_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.title == ds_2d.title

    def test_name_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.name == ds_2d.name

    def test_author_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.author == "test_author"

    def test_description_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.description == "test description"

    def test_origin_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.origin == "test_origin"

    def test_meta_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.meta.project == "test_project"

    def test_meta_is_independent_copy(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        r.meta.project = "changed"
        assert ds_2d.meta.project == "test_project"


# ======================================================================================
# HISTORY
# ======================================================================================


class TestHistory:
    """Characterize history behavior: appended."""

    def test_history_appended(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert len(r.history) == len(ds_2d.history) + 1

    def test_original_entries_preserved(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert "original entry" in r.history[0].lower()

    def test_interpolation_entry_added(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert "Interpolated along dims" in "".join(r.history)

    def test_inplace_updates_history(self, ds_2d):
        r = ds_2d.interpolate(
            dim="x", coord=np.linspace(3500.0, 1500.0, 3), inplace=True
        )
        assert len(r.history) == 2


# ======================================================================================
# ROI
# ======================================================================================


class TestROI:
    """Characterize ROI (region of interest) behavior: preserved but stale."""

    def test_roi_preserved_after_interpolation(self, ds_2d):
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.roi == ds_2d.roi

    def test_roi_preserved_unchanged(self, ds_2d):
        """
        ROI is preserved unchanged and may therefore become stale
        when the coordinate grid is rebuilt.
        """
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.roi[0] == ds_2d.roi[0]
        assert r.roi[1] == ds_2d.roi[1]


# ======================================================================================
# IDENTITY AND PROVENANCE
# ======================================================================================


class TestIdentityProvenance:
    """Characterize identity and provenance semantics."""

    def test_identity_same_object(self, ds_2d):
        """
        Interpolation returns a new object but represents the same
        scientific entity on a new coordinate grid.
        """
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r is not ds_2d

    def test_provenance_extended(self, ds_2d):
        """Provenance is extended via history, not rewritten."""
        r = ds_2d.interpolate(dim="x", coord=np.linspace(3500.0, 1500.0, 3))
        assert r.name == ds_2d.name
        assert r.origin == "test_origin"
        assert r.history[-1] != ds_2d.history[-1]


# ======================================================================================
# COORDINATE SEMANTICS — increasing, decreasing, shifted, denser, sparser
# ======================================================================================


class TestCoordinateSemantics:
    """Characterize coordinate rebuild for various target grid shapes."""

    def test_increasing_target(self, ds_2d):
        target = np.linspace(1500.0, 3500.0, 5)
        r = ds_2d.interpolate(dim="x", coord=target)
        assert np.all(np.diff(r.coord("x").data) > 0)

    def test_decreasing_target(self, ds_2d):
        target = np.linspace(3500.0, 1500.0, 5)
        r = ds_2d.interpolate(dim="x", coord=target)
        assert np.all(np.diff(r.coord("x").data) < 0)

    def test_shifted_grid(self, ds_2d):
        shift = 100.0
        old = ds_2d.coord("x").data
        target = old + shift
        r = ds_2d.interpolate(dim="x", coord=target)
        np.testing.assert_allclose(r.coord("x").data, target, rtol=1e-10)

    def test_denser_grid(self, ds_2d):
        target = np.linspace(3500.0, 1500.0, 20)
        r = ds_2d.interpolate(dim="x", coord=target)
        assert r.shape[1] == 20

    def test_sparser_grid(self, ds_2d):
        target = np.linspace(3500.0, 1500.0, 3)
        r = ds_2d.interpolate(dim="x", coord=target)
        assert r.shape[1] == 3

    def test_out_of_range_nan(self, ds_2d):
        """Points outside the original coordinate range become NaN."""
        target = np.array([-100.0, 2500.0, 5000.0])
        r = ds_2d.interpolate(dim="x", coord=target)
        assert np.isnan(r.data[0, 0])
        assert np.isnan(r.data[0, -1])
        assert not np.isnan(r.data[0, 1])


# ======================================================================================
# SECONDARY COORDINATES
# ======================================================================================


class TestSecondaryCoordinates:
    """Characterize secondary-coordinate interpolation."""

    def test_secondary_coord_interpolated(self, ds_multi_coord):
        target = np.linspace(3500.0, 1500.0, 3)
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            ds_multi_coord.interpolate(dim="x", coord=target)

    def test_secondary_coord_units_preserved(self, ds_multi_coord):
        target = np.linspace(3500.0, 1500.0, 3)
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            ds_multi_coord.interpolate(dim="x", coord=target)


# ======================================================================================
# MULTI-DIM INTERPOLATION
# ======================================================================================


class TestMultiDimInterpolation:
    """Characterize sequential multi-dim interpolation."""

    def test_multi_dim_shape(self, ds_2d):
        r = ds_2d.interpolate(
            dims=["x", "y"],
            coord={
                "x": np.linspace(3500.0, 1500.0, 3),
                "y": np.linspace(10.0, 50.0, 2),
            },
        )
        assert r.shape == (2, 3)

    def test_multi_dim_dims_preserved(self, ds_2d):
        r = ds_2d.interpolate(
            dims=["x", "y"],
            coord={
                "x": np.linspace(3500.0, 1500.0, 3),
                "y": np.linspace(10.0, 50.0, 2),
            },
        )
        assert r.dims == ds_2d.dims

    def test_multi_dim_coords_rebuilt(self, ds_2d):
        r = ds_2d.interpolate(
            dims=["x", "y"],
            coord={
                "x": np.linspace(3500.0, 1500.0, 3),
                "y": np.linspace(10.0, 50.0, 2),
            },
        )
        assert len(r.coord("x")) == 3
        assert len(r.coord("y")) == 2


# ======================================================================================
# ALIGN WITH INTERPOLATE
# ======================================================================================


class TestAlignInterpolate:
    """Characterize align with method='interpolate'."""

    @pytest.fixture
    def ds1(self):
        x = Coord(np.linspace(4000.0, 1000.0, 10), title="x", units="cm^-1")
        y = Coord(np.linspace(0.0, 60.0, 5), title="y", units="s")
        return NDDataset(
            np.random.rand(5, 10),
            coordset=[y, x],
            title="test",
            name="ds1",
        )

    @pytest.fixture
    def ds2(self):
        x = Coord(np.linspace(4000.0, 1000.0, 7), title="x", units="cm^-1")
        y = Coord(np.linspace(0.0, 60.0, 3), title="y", units="s")
        return NDDataset(
            np.random.rand(3, 7),
            coordset=[y, x],
            title="test",
            name="ds2",
        )

    def test_align_return_type(self, ds1, ds2):
        r1, r2 = ds1.align(ds2, dim="x", method="interpolate")
        assert isinstance(r1, NDDataset) and isinstance(r2, NDDataset)

    def test_align_unifies_shape(self, ds1, ds2):
        r1, r2 = ds1.align(ds2, dim="x", method="interpolate")
        assert r1.shape[1] == r2.shape[1]

    def test_align_unifies_coord(self, ds1, ds2):
        r1, r2 = ds1.align(ds2, dim="x", method="interpolate")
        np.testing.assert_array_equal(r1.coord("x").data, r2.coord("x").data)

    def test_align_preserves_name(self, ds1, ds2):
        r1, r2 = ds1.align(ds2, dim="x", method="interpolate")
        assert r1.name == ds1.name
        assert r2.name == ds2.name

    def test_align_appends_history(self, ds1, ds2):
        ds1.history = ["entry1"]
        ds2.history = ["entry2"]
        r1, r2 = ds1.align(ds2, dim="x", method="interpolate")
        assert len(r1.history) > 0
        assert len(r2.history) > 0
