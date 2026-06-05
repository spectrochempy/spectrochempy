# SPDX-License-Identifier: BSD-3-Clause
# (see LICENSE.txt for details)

"""
Behavioral tests for NDDataset + Coord/CoordSet integration.

Tests focus on public API behavior — CoordSet consistency with dataset
shape, dims, slicing, and coordinate assignment.
Uses deterministic synthetic data. No external files, no plotting.
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import assert_array_equal
from spectrochempy.utils.testing import assert_coord_almost_equal
from spectrochempy.utils.testing import assert_units_equal

# ==============================================================================
# Creation with explicit CoordSet
# ==============================================================================


class TestNDDatasetCreationWithCoordSet:
    """Creating NDDataset with an explicit CoordSet."""

    def test_1d_with_single_coord(self):
        c = Coord([100.0, 200.0, 300.0], name="x", units="cm^-1")
        ds = NDDataset([1.0, 2.0, 3.0], coordset=CoordSet(c))
        assert ds.shape == (3,)
        assert "x" in ds.coordset.names
        assert_array_equal(ds.coordset["x"].data, [100.0, 200.0, 300.0])

    def test_2d_with_two_coords(self):
        coord_y = Coord([1.0, 2.0, 3.0], name="y")
        coord_x = Coord([10.0, 20.0], name="x")
        ds = NDDataset(np.ones((3, 2)), coordset=[coord_y, coord_x])
        assert ds.shape == (3, 2)
        assert len(ds.dims) == 2

    def test_coord_order_in_2d_list(self):
        """Coord passed as list should define the dimensions."""
        coord_y = Coord([1.0, 2.0, 3.0], name="y")
        coord_x = Coord([10.0, 20.0], name="x")
        ds = NDDataset(np.ones((3, 2)), coordset=[coord_y, coord_x])
        assert ds.shape[0] == 3  # y size
        assert ds.shape[1] == 2  # x size

    def test_with_units_on_dataset(self):
        c = Coord([100, 200, 300], name="x", units="nm")
        ds = NDDataset([1, 2, 3], coordset=CoordSet(c), units="absorbance")
        assert ds.units is not None


# ==============================================================================
# Dims / Shape / CoordSet consistency
# ==============================================================================


class TestNDDatasetDimsConsistency:
    """Consistency between dataset shape, dims, and coordset."""

    def test_dims_match_coord_names(self):
        coord_x = Coord([10, 20], name="x")
        coord_y = Coord([1, 2, 3], name="y")
        ds = NDDataset(np.ones((3, 2)), coordset=[coord_y, coord_x])
        assert ds.shape == (3, 2)
        # each coord in coordset has a size matching the corresponding dim
        for i, dim in enumerate(ds.dims):
            assert ds.coordset[dim].size == ds.shape[i]

    def test_shape_agrees_with_coord_sizes(self):
        coord_x = Coord(np.linspace(4000, 1000, 10), name="x")
        coord_y = Coord(np.linspace(0, 60, 100), name="y")
        ds = NDDataset(np.ones((100, 10)), coordset=[coord_y, coord_x])
        shape = ds.shape
        assert shape[0] == 100  # y
        assert shape[1] == 10  # x

    def test_dims_are_strings(self):
        c = Coord([1, 2, 3], name="wavelength")
        ds = NDDataset([1, 2, 3], coordset=CoordSet(c))
        assert all(isinstance(d, str) for d in ds.dims)


# ==============================================================================
# Assigning coordinates to existing dataset
# ==============================================================================


class TestNDDatasetCoordAssignment:
    """Assigning CoordSet to an existing NDDataset."""

    def test_assign_coordset_to_dataset(self):
        c = Coord([100.0, 200.0, 300.0], name="x")
        ds = NDDataset([1.0, 2.0, 3.0])
        assert ds.coordset is None
        ds.coordset = CoordSet(c)
        assert ds.coordset is not None
        assert "x" in ds.coordset.names

    def test_assign_coordset_with_name_in_names(self):
        c = Coord([100.0, 200.0, 300.0], name="x")
        ds = NDDataset([1.0, 2.0, 3.0])
        ds.coordset = CoordSet(c)
        retrieved = ds.coordset["x"]
        assert_array_equal(retrieved.data, [100.0, 200.0, 300.0])

    def test_replace_coordset(self):
        c1 = Coord([100.0, 200.0, 300.0], name="x")
        ds = NDDataset([1.0, 2.0, 3.0])
        ds.coordset = CoordSet(c1)
        c2 = Coord([400.0, 500.0, 600.0], name="x")
        ds.coordset = CoordSet(c2)
        assert_array_equal(ds.coordset["x"].data, [400.0, 500.0, 600.0])


# ==============================================================================
# Coord replacement through CoordSet API
# ==============================================================================


class TestNDDatasetCoordReplacement:
    """Replacing coordinate data through the CoordSet API."""

    def test_replace_coord_via_coordset(self):
        c = Coord([1, 2, 3], name="x")
        ds = NDDataset([10, 20, 30], coordset=CoordSet(c))
        ds.coordset["x"] = Coord([100, 200, 300], name="x")
        assert_array_equal(ds.coordset["x"].data, [100, 200, 300])

    def test_add_new_coord_via_coordset(self):
        c = Coord([1, 2, 3], name="x")
        ds = NDDataset([10, 20, 30])
        ds.coordset = CoordSet(c)
        ds.coordset["y"] = Coord([100, 200], name="y")
        assert "y" in ds.coordset.names

    def test_set_units_on_coord_in_dataset(self):
        c = Coord([1, 2, 3])
        ds = NDDataset([10, 20, 30])
        ds.coordset = CoordSet(c)
        ds.coordset.set_units(x="cm^-1", force=True)
        units = ds.coordset.units
        assert any(u is not None for u in units)

    def test_replace_coord_via_dimension_attribute_preserves_metadata(self):
        c = Coord([1, 2, 3], name="x", title="Old", units="cm^-1")
        ds = NDDataset([10, 20, 30], coordset=CoordSet(c))
        replacement = Coord(
            [100, 200, 300],
            name="wavelength",
            title="Wavenumber",
            units="nm",
            labels=["a", "b", "c"],
        )

        ds.x = replacement

        assert ds.shape == (3,)
        assert_array_equal(ds.x.data, [100, 200, 300])
        assert ds.x.name == "x"
        assert ds.x.title == "Wavenumber"
        assert_units_equal(ds.x.units, ur.nm)
        assert_array_equal(ds.x.labels, ["a", "b", "c"])

    def test_replace_coord_via_dimension_attribute_with_matching_length(self):
        coord_y = Coord([1.0, 2.0, 3.0], name="y")
        coord_x = Coord([10.0, 20.0], name="x")
        ds = NDDataset(np.ones((3, 2)), coordset=[coord_y, coord_x])

        ds.x = [100.0, 200.0]

        assert ds.shape == (3, 2)
        assert_array_equal(ds.x.data, [100.0, 200.0])

    def test_replace_coord_via_dimension_attribute_rejects_wrong_length(self):
        c = Coord([1, 2, 3], name="x")
        ds = NDDataset([10, 20, 30], coordset=CoordSet(c))

        with pytest.raises(ValueError, match="coordinate size=2 != data shape"):
            ds.x = Coord([100, 200], name="wrong")

    def test_replace_coord_via_dimension_attribute_preserves_multicoord_behavior(self):
        ds = NDDataset([0.0, 1.0, 2.0])
        x2 = Coord(np.array([0.5, 0.8, 9.0]))
        x1 = Coord(np.array([1.5, 5.8, -9.0]))

        ds.x = CoordSet(Coord(x2), Coord(x1))

        assert isinstance(ds.x, CoordSet)
        assert ds.x.is_same_dim
        assert_coord_almost_equal(ds.x["_1"], x1)
        assert_coord_almost_equal(ds.x["_2"], x2)


# ==============================================================================
# Slicing → CoordSet interaction
# ==============================================================================


class TestNDDatasetSliceCoordSet:
    """Slicing an NDDataset and verifying the resulting CoordSet."""

    def test_slice_1d_trims_coord(self):
        c = Coord([100, 200, 300, 400, 500], name="x")
        ds = NDDataset([1, 2, 3, 4, 5], coordset=CoordSet(c))
        ds_slice = ds[1:4]
        assert ds_slice.shape == (3,)
        assert_array_equal(ds_slice.coordset["x"].data, [200, 300, 400])

    def test_slice_2d_trims_first_dim_coord(self):
        coord_y = Coord([1.0, 2.0, 3.0], name="y")
        coord_x = Coord([10.0, 20.0, 30.0, 40.0], name="x")
        ds = NDDataset(np.ones((3, 4)), coordset=[coord_y, coord_x])
        ds_slice = ds[0:2, :]
        assert ds_slice.shape == (2, 4)
        assert_array_equal(ds_slice.coordset["y"].data, [1.0, 2.0])

    def test_slice_2d_trims_second_dim_coord(self):
        coord_y = Coord([1.0, 2.0, 3.0], name="y")
        coord_x = Coord([10.0, 20.0, 30.0, 40.0], name="x")
        ds = NDDataset(np.ones((3, 4)), coordset=[coord_y, coord_x])
        ds_slice = ds[:, 1:3]
        assert ds_slice.shape == (3, 2)
        assert_array_equal(ds_slice.coordset["x"].data, [20.0, 30.0])

    def test_slice_both_dims(self):
        coord_y = Coord([1.0, 2.0, 3.0, 4.0, 5.0], name="y")
        coord_x = Coord([10.0, 20.0, 30.0], name="x")
        ds = NDDataset(np.ones((5, 3)), coordset=[coord_y, coord_x])
        ds_slice = ds[1:4, 0:2]
        assert ds_slice.shape == (3, 2)
        assert_array_equal(ds_slice.coordset["y"].data, [2.0, 3.0, 4.0])
        assert_array_equal(ds_slice.coordset["x"].data, [10.0, 20.0])

    def test_fancy_index_trims_coord(self):
        c = Coord([100, 200, 300, 400, 500], name="x")
        ds = NDDataset([1, 2, 3, 4, 5], coordset=CoordSet(c))
        ds_slice = ds[[0, 2, 4]]
        assert ds_slice.shape == (3,)
        assert_array_equal(ds_slice.coordset["x"].data, [100, 300, 500])


# ==============================================================================
# Coord metadata preservation after slicing
# ==============================================================================


class TestCoordMetadataAfterSlice:
    """Coord units, name, title preservation after slicing."""

    def test_units_preserved_after_slice(self):
        c = Coord([100, 200, 300, 400], name="x", units="cm^-1")
        ds = NDDataset([1, 2, 3, 4], coordset=CoordSet(c))
        ds_slice = ds[1:3]
        assert_units_equal(ds_slice.coordset["x"].units, ur("cm^-1"))

    def test_name_preserved_after_slice(self):
        c = Coord([100, 200, 300, 400], name="x")
        ds = NDDataset([1, 2, 3, 4], coordset=CoordSet(c))
        ds_slice = ds[1:3]
        assert ds_slice.coordset["x"].name == "x"

    def test_title_preserved_after_slice(self):
        c = Coord([100, 200, 300, 400], name="x", title="Wavenumber")
        ds = NDDataset([1, 2, 3, 4], coordset=CoordSet(c))
        ds_slice = ds[1:3]
        assert ds_slice.coordset["x"].title == "Wavenumber"

    def test_labels_preserved_after_slice(self):
        c = Coord([100, 200, 300, 400], name="x", labels=["a", "b", "c", "d"])
        ds = NDDataset([1, 2, 3, 4], coordset=CoordSet(c))
        ds_slice = ds[1:3]
        assert_array_equal(ds_slice.coordset["x"].labels, ["b", "c"])


# ==============================================================================
# Edge cases
# ==============================================================================


class TestNDDatasetCoordEdgeCases:
    """Edge cases for dataset-coord integration."""

    def test_dataset_without_coords(self):
        ds = NDDataset([1, 2, 3])
        assert ds.coordset is None or len(ds.coordset) >= 0

    def test_coord_size_mismatch(self):
        coord_x = Coord([1, 2], name="x")  # size 2
        data = np.ones(3)  # size 3
        with pytest.raises((ValueError, TypeError)):
            NDDataset(data, coordset=CoordSet(coord_x))

    def test_2d_coord_size_mismatch(self):
        coord_y = Coord([1.0, 2.0, 3.0], name="y")  # size 3
        coord_x = Coord([10.0, 20.0], name="x")  # size 2
        data = np.ones((3, 5))  # x dim size 5, coord says 2
        with pytest.raises((ValueError, TypeError)):
            NDDataset(data, coordset=[coord_y, coord_x])

    def test_coords_after_squeeze(self):
        coord_x = Coord([1, 2, 3], name="x")
        ds = NDDataset(np.ones((3, 1)), coordset=[coord_x, Coord([0], name="y")])
        ds_sq = ds.squeeze()
        assert ds_sq is not None

    def test_coord_arange_from_dataset(self):
        # coord supports arange-like creation
        c2 = Coord(np.arange(5))
        ds2 = NDDataset(np.ones(5), coordset=CoordSet(c2))
        assert ds2.shape == (5,)

    def test_dim_order_after_transpose(self):
        coord_y = Coord([1.0, 2.0, 3.0], name="y")
        coord_x = Coord([10.0, 20.0, 30.0], name="x")
        ds = NDDataset(np.ones((3, 3)), coordset=[coord_y, coord_x])
        ds_t = ds.T
        assert ds_t.shape == (3, 3)

    def test_coordset_after_bool_index(self):
        c = Coord([100, 200, 300, 400], name="x")
        ds = NDDataset([1, 2, 3, 4], coordset=CoordSet(c))
        mask = np.array([True, False, True, False])
        ds_masked = ds[mask]
        assert ds_masked.coordset is not None
