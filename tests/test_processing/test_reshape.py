# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for NDDataset.reshape()."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset


class TestNDDatasetReshape:
    """Test suite for NDDataset.reshape()."""

    def test_reshape_matches_numpy(self):
        """Reshaped data matches numpy.reshape."""
        data = np.arange(24).reshape(4, 6)
        ds = NDDataset(data)
        reshaped = ds.reshape((2, 3, 4))
        np.testing.assert_array_equal(reshaped.data, data.reshape(2, 3, 4))

    def test_reshape_negative_one_inference(self):
        """-1 dimension is inferred correctly."""
        data = np.arange(24).reshape(4, 6)
        ds = NDDataset(data)
        reshaped = ds.reshape((2, -1, 4))
        assert reshaped.shape == (2, 3, 4)
        np.testing.assert_array_equal(reshaped.data, data.reshape(2, 3, 4))

    def test_reshape_invalid_negative_one(self):
        """Multiple -1 dimensions raise ValueError."""
        ds = NDDataset(np.arange(24).reshape(4, 6))
        with pytest.raises(ValueError, match="Only one dimension can be -1"):
            ds.reshape((-1, -1, 4))

    def test_reshape_size_mismatch(self):
        """Incompatible shapes raise ValueError."""
        ds = NDDataset(np.arange(24).reshape(4, 6))
        with pytest.raises(ValueError, match="Cannot reshape array"):
            ds.reshape((5, 5))

    def test_reshape_preserves_units_title(self):
        """Units, title, and name are preserved."""
        ds = NDDataset(np.random.rand(4, 6))
        ds.units = "K"
        ds.title = "temperature"
        ds.name = "temp_ds"
        reshaped = ds.reshape((2, 3, 4))
        # Units are preserved (pint may normalize 'K' to 'K' or 'kelvin')
        assert reshaped.units.dimensionality == ds.units.dimensionality
        assert reshaped.title == "temperature"
        assert reshaped.name == "temp_ds"

    def test_reshape_noop_preserves_dims_and_coords(self):
        """No-op reshape preserves dims and coords."""
        ds = NDDataset(np.random.rand(4, 6))
        ds.set_coordset(
            y=Coord(np.arange(4), title="y"),
            x=Coord(np.arange(6), title="x"),
        )
        reshaped = ds.reshape((4, 6))
        assert reshaped.dims == ["y", "x"]
        np.testing.assert_array_equal(reshaped.coordset.y.data, np.arange(4))
        np.testing.assert_array_equal(reshaped.coordset.x.data, np.arange(6))

    def test_reshape_add_singleton_dimension(self):
        """Adding a singleton dimension preserves unambiguous coords."""
        ds = NDDataset(np.random.rand(4, 6))
        ds.set_coordset(
            y=Coord(np.arange(4), title="y"),
            x=Coord(np.arange(6), title="x"),
        )
        reshaped = ds.reshape((1, 4, 6), dims=("u", "y", "x"))
        assert reshaped.shape == (1, 4, 6)
        assert reshaped.dims == ["u", "y", "x"]
        # y and x coords preserved because shape matches unambiguously
        np.testing.assert_array_equal(reshaped.coordset.y.data, np.arange(4))
        np.testing.assert_array_equal(reshaped.coordset.x.data, np.arange(6))

    def test_reshape_split_dimension_safe_drops_ambiguous(self):
        """Splitting a dimension with safe policy drops ambiguous coords."""
        ds = NDDataset(np.random.rand(120, 1000))
        ds.set_coordset(
            y=Coord(np.arange(120), title="time"),
            x=Coord(np.arange(1000), title="wavenumber"),
        )
        reshaped = ds.reshape((2, 60, 1000), dims=("cycle", "time", "x"))
        assert reshaped.shape == (2, 60, 1000)
        # x coord preserved (shape 1000 unchanged)
        np.testing.assert_array_equal(reshaped.coordset.x.data, np.arange(1000))
        # y (was 120) split into cycle (2) and time (60) — ambiguous, dropped
        # Empty coords may exist but have no meaningful data
        assert reshaped.coordset.x.title == "wavenumber"

    def test_reshape_merge_dimensions_safe_drops_ambiguous(self):
        """Merging dimensions with safe policy drops ambiguous coords."""
        ds = NDDataset(np.random.rand(2, 60, 1000))
        ds.set_coordset(
            z=Coord(np.arange(2), title="cycle"),
            y=Coord(np.arange(60), title="time"),
            x=Coord(np.arange(1000), title="wavenumber"),
        )
        reshaped = ds.reshape((120, 1000), dims=("time", "x"))
        assert reshaped.shape == (120, 1000)
        # x preserved
        np.testing.assert_array_equal(reshaped.coordset.x.data, np.arange(1000))

    def test_reshape_explicit_dims(self):
        """Explicit dims are applied and validated."""
        ds = NDDataset(np.random.rand(4, 6))
        reshaped = ds.reshape((2, 3, 4), dims=("a", "b", "c"))
        assert reshaped.dims == ["a", "b", "c"]

    def test_reshape_explicit_dims_wrong_length(self):
        """Explicit dims with wrong length raise ValueError."""
        ds = NDDataset(np.random.rand(4, 6))
        with pytest.raises(ValueError, match="dims length"):
            ds.reshape((2, 3, 4), dims=("a", "b"))

    def test_reshape_explicit_dims_not_unique(self):
        """Non-unique explicit dims raise ValueError."""
        ds = NDDataset(np.random.rand(4, 6))
        with pytest.raises(ValueError, match="dims must be unique"):
            ds.reshape((2, 3, 4), dims=("a", "a", "c"))

    def test_reshape_explicit_coords(self):
        """User-provided coords override inferred ones."""
        ds = NDDataset(np.random.rand(120, 1000))
        ds.set_coordset(
            y=Coord(np.arange(120), title="time"),
            x=Coord(np.arange(1000), title="wavenumber"),
        )
        reshaped = ds.reshape(
            (2, 60, 1000),
            dims=("cycle", "time", "x"),
            coords={"cycle": Coord([10, 20], title="cycle_idx")},
        )
        np.testing.assert_array_equal(reshaped.coordset.cycle.data, [10, 20])
        assert reshaped.coordset.cycle.title == "cycle_idx"

    def test_reshape_explicit_coords_validation(self):
        """Explicit coords with wrong length raise ValueError."""
        ds = NDDataset(np.random.rand(120, 1000))
        with pytest.raises(ValueError, match="length"):
            ds.reshape(
                (2, 60, 1000),
                dims=("cycle", "time", "x"),
                coords={"cycle": Coord([0, 1, 2])},  # length 3 != 2
            )

    def test_reshape_explicit_coords_unknown_dim(self):
        """Explicit coords for unknown dim raise ValueError."""
        ds = NDDataset(np.random.rand(120, 1000))
        with pytest.raises(ValueError, match="not found in new dims"):
            ds.reshape(
                (2, 60, 1000),
                dims=("cycle", "time", "x"),
                coords={"unknown": Coord([0, 1])},
            )

    def test_reshape_coord_policy_strict_unambiguous(self):
        """Strict mode succeeds when all coords map unambiguously."""
        ds = NDDataset(np.random.rand(4, 6))
        ds.set_coordset(
            y=Coord(np.arange(4)),
            x=Coord(np.arange(6)),
        )
        reshaped = ds.reshape((4, 6), dims=("y", "x"), coord_policy="strict")
        np.testing.assert_array_equal(reshaped.coordset.y.data, np.arange(4))
        np.testing.assert_array_equal(reshaped.coordset.x.data, np.arange(6))

    def test_reshape_coord_policy_strict_ambiguous_size(self):
        """Strict mode raises when size mapping is ambiguous."""
        ds = NDDataset(np.random.rand(4, 4))
        ds.set_coordset(
            y=Coord(np.arange(4)),
            x=Coord(np.arange(4)),
        )
        with pytest.raises(ValueError, match="strict mode"):
            ds.reshape((2, 8), dims=("y", "x"), coord_policy="strict")

    def test_reshape_coord_policy_strict_renamed(self):
        """Strict mode raises when dim names change."""
        ds = NDDataset(np.random.rand(4, 6))
        ds.set_coordset(
            y=Coord(np.arange(4)),
            x=Coord(np.arange(6)),
        )
        with pytest.raises(ValueError, match="strict mode"):
            ds.reshape((4, 6), dims=("a", "b"), coord_policy="strict")

    def test_reshape_coord_policy_drop(self):
        """Drop mode discards all old coordinates."""
        ds = NDDataset(np.random.rand(4, 6))
        ds.set_coordset(
            y=Coord(np.arange(4)),
            x=Coord(np.arange(6)),
        )
        reshaped = ds.reshape((2, 3, 4), coord_policy="drop")
        assert reshaped.coordset is None

    def test_reshape_history(self):
        """History records the reshape operation."""
        ds = NDDataset(np.random.rand(4, 6))
        reshaped = ds.reshape((2, 3, 4))
        assert "reshaped from (4, 6) to (2, 3, 4)" in reshaped.history[-1].lower()

    def test_reshape_inplace(self):
        """inplace=True modifies the original object."""
        ds = NDDataset(np.random.rand(4, 6))
        original_id = id(ds)
        result = ds.reshape((2, 3, 4), inplace=True)
        assert id(result) == original_id
        assert result.shape == (2, 3, 4)

    def test_reshape_top_level_api(self):
        """scp.reshape(X, shape) works as a top-level helper."""
        ds = NDDataset(np.random.rand(4, 6))
        result = scp.reshape(ds, (2, 3, 4))
        assert result.shape == (2, 3, 4)

    def test_reshape_mask(self):
        """Mask is reshaped together with data."""
        data = np.arange(24).reshape(4, 6)
        ds = NDDataset(data)
        ds.mask = data > 10
        reshaped = ds.reshape((2, 3, 4))
        expected_mask = (data > 10).reshape(2, 3, 4)
        np.testing.assert_array_equal(reshaped.mask, expected_mask)
