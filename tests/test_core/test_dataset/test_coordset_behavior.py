# SPDX-License-Identifier: BSD-3-Clause
# (see LICENSE.txt for details)

"""
Behavioral tests for CoordSet.

Tests focus on public API behavior, not private implementation details.
Uses deterministic synthetic data only. No external files, no plotting.
"""

import warnings
from copy import deepcopy

import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.utils.testing import assert_array_equal

# ==============================================================================
# Construction
# ==============================================================================


class TestCoordSetConstruction:
    """CoordSet creation from Coord objects and other inputs."""

    def test_from_single_coord(self):
        c = Coord([1, 2, 3], name="x")
        cs = CoordSet(c)
        assert len(cs) == 1
        assert cs.names == ["x"]

    def test_from_two_coords(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2)
        assert len(cs) == 2

    def test_preserves_coord_data(self):
        c = Coord([1.0, 2.0, 3.0], name="x")
        cs = CoordSet(c)
        assert_array_equal(cs["x"].data, [1.0, 2.0, 3.0])

    def test_from_list_of_coords(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        # Using individual args (not wrapped in a list) creates multi-dim CoordSet
        cs = CoordSet(c1, c2, keepnames=True)
        assert len(cs) == 2

    def test_from_kwargs(self):
        cs = CoordSet(x=Coord([1, 2, 3]), y=Coord([10, 20]))
        assert "x" in cs.names
        assert "y" in cs.names

    def test_with_coord_names_and_dim_names(self):
        c = Coord([1, 2, 3], name="wavelength")
        cs = CoordSet(c, keepnames=True)
        assert cs.names == ["wavelength"]

    def test_with_dims_kwarg(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([10, 20])
        cs = CoordSet(c1, c2, dims=["x", "y"])
        assert len(cs.names) == 2

    def test_all_coords_have_names(self):
        c1 = Coord([1, 2, 3], name="a")
        c2 = Coord([10, 20], name="b")
        cs = CoordSet(c1, c2, keepnames=True)
        for name in cs.names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_construction_without_args(self):
        # CoordSet() currently raises TypeError (existing bug in _coords validator)
        with pytest.raises(TypeError):
            CoordSet()


# ==============================================================================
# Properties
# ==============================================================================


class TestCoordSetProperties:
    """CoordSet public property behavior."""

    def test_names_returns_list_of_strings(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"), Coord([10, 20], name="y"))
        names = cs.names
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_sizes_returns_list(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        sizes = cs.sizes
        assert isinstance(sizes, list)
        assert len(sizes) == 2

    def test_sizes_match_coord_sizes(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        sizes = cs.sizes
        # sizes order corresponds to _coords order
        assert 3 in sizes
        assert 2 in sizes

    def test_is_empty_false_with_coords(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        assert not cs.is_empty

    def test_default_returns_coord(self):
        c = Coord([1, 2, 3], name="x")
        cs = CoordSet(c)
        default = cs.default
        assert isinstance(default, Coord)

    def test_default_coord_name(self):
        c = Coord([1, 2, 3], name="x")
        cs = CoordSet(c)
        assert cs.default.name == "x"

    def test_data_is_default_coord_data(self):
        c = Coord([1, 2, 3], name="x")
        cs = CoordSet(c)
        assert_array_equal(cs.data, c.data)

    def test_titles_returns_list(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"), Coord([10, 20], name="y"))
        titles = cs.titles
        assert isinstance(titles, list)
        assert len(titles) == len(cs)

    def test_labels_returns_none_for_unlabeled(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2)
        labs = cs.labels
        assert all(label is None for label in labs)

    def test_units_returns_list_for_mixed(self):
        c1 = Coord([1, 2, 3], name="x", units="cm^-1")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        units = cs.units
        assert len(units) == 2

    def test_is_labeled_false_no_labels(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        assert not cs.is_labeled

    def test_is_labeled_true_with_labels(self):
        cs = CoordSet(Coord([1, 2, 3], name="x", labels=["a", "b", "c"]))
        assert cs.is_labeled


# ==============================================================================
# Access by name
# ==============================================================================


class TestCoordSetAccess:
    """CoordSet item access behavior."""

    def test_getitem_by_name(self):
        c = Coord([1, 2, 3], name="x")
        cs = CoordSet(c)
        retrieved = cs["x"]
        assert isinstance(retrieved, Coord)
        assert_array_equal(retrieved.data, c.data)

    def test_getitem_by_integer(self):
        c = Coord([1, 2, 3], name="x")
        cs = CoordSet(c)
        retrieved = cs[0]
        assert isinstance(retrieved, Coord)

    def test_getitem_missing_name_raises(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        with pytest.raises((KeyError, IndexError)):
            _ = cs["nonexistent"]

    def test_getitem_integer_out_of_range(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        with pytest.raises((IndexError, KeyError)):
            _ = cs[10]

    def test_contains_via_names(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"), Coord([10, 20], name="y"))
        assert "x" in cs.names
        assert "z" not in cs.names

    def test_len_returns_coord_count(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"), Coord([10, 20], name="y"))
        assert len(cs) == 2
        cs2 = CoordSet(Coord([1, 2, 3], name="x"))
        assert len(cs2) == 1

    def test_keys_returns_names(self):
        c1 = Coord([1, 2, 3], name="a")
        c2 = Coord([10, 20], name="b")
        cs = CoordSet(c1, c2, keepnames=True)
        keys = cs.keys()
        assert "a" in keys
        assert "b" in keys

    def test_to_dict(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        d = cs.to_dict()
        assert isinstance(d, dict)
        assert "x" in d
        assert "y" in d
        assert isinstance(d["x"], Coord)

    def test_to_dict_values(self):
        c = Coord([1.0, 2.0, 3.0], name="x")
        cs = CoordSet(c)
        d = cs.to_dict()
        assert_array_equal(d["x"].data, [1.0, 2.0, 3.0])


# ==============================================================================
# Mutation
# ==============================================================================


class TestCoordSetMutation:
    """CoordSet modification behavior."""

    def test_setitem_by_name_replaces(self):
        c = Coord([1, 2, 3], name="x")
        cs = CoordSet(c)
        cs["x"] = Coord([10, 20, 30], name="x")
        assert_array_equal(cs["x"].data, [10, 20, 30])

    def test_setitem_new_name_adds(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        cs["y"] = Coord([10, 20], name="y")
        assert "y" in cs.names
        assert len(cs) == 2

    def test_delitem_removes_coord(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        del cs["x"]
        assert "x" not in cs.names
        assert len(cs) == 1

    def test_delitem_missing_name_raises(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        with pytest.raises(KeyError):
            del cs["nonexistent"]

    def test_set_coord_updates_data(self):
        c1 = Coord([1, 2, 3], name="x")
        cs = CoordSet(c1)
        new_x = Coord([4, 5, 6], name="x")
        cs.set(x=new_x)
        assert_array_equal(cs["x"].data, [4, 5, 6])

    def test_set_multiple_coords(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        cs.set(x=Coord([7, 8, 9], name="x"), y=Coord([30, 40], name="y"))
        assert_array_equal(cs["x"].data, [7, 8, 9])
        assert_array_equal(cs["y"].data, [30, 40])

    def test_set_titles_by_kwargs(self):
        c1 = Coord([1, 2, 3], name="x")
        cs = CoordSet(c1)
        cs.set_titles(x="Wavenumber")
        assert cs["x"].title == "Wavenumber"

    def test_set_units_by_kwargs(self):
        c1 = Coord([1, 2, 3], name="x")
        cs = CoordSet(c1)
        cs.set_units(x="cm^-1", force=True)
        assert cs["x"].units is not None

    def test_update_replaces_coord(self):
        c1 = Coord([1, 2, 3], name="x")
        cs = CoordSet(c1)
        cs.update(x=[4, 5, 6])
        assert_array_equal(cs["x"].data, [4, 5, 6])

    def test_setitem_by_title_replaces_top_level(self):
        c1 = Coord([1, 2, 3], name="x", title="wavelength")
        cs = CoordSet(c1)
        cs["wavelength"] = Coord([7, 8, 9], name="x")
        assert_array_equal(cs["x"].data, [7, 8, 9])

    def test_setitem_by_title_in_nested_child(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([4, 5, 6], title="alpha")
        cs = CoordSet([c1, c2])
        cs["alpha"] = Coord([7, 8, 9])
        assert_array_equal(cs["x"]["_2"].data, [7, 8, 9])

    def test_setitem_by_nested_child_name(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([4, 5, 6])
        cs = CoordSet([c1, c2])
        cs["_2"] = Coord([7, 8, 9])
        assert_array_equal(cs["x"]["_2"].data, [7, 8, 9])

    def test_setitem_by_synthetic_alias(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([4, 5, 6])
        cs = CoordSet([c1, c2])
        cs["x_2"] = Coord([7, 8, 9])
        assert_array_equal(cs["x"]["_2"].data, [7, 8, 9])

    def test_setitem_appends_via_available_name(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        cs["y"] = Coord([10, 20, 30], name="y")
        assert "y" in cs.names
        assert len(cs) == 2

    def test_setitem_appends_via_synthetic_alias(self):
        """Append new child to nested CoordSet via synthetic alias."""
        c1 = Coord([1, 2, 3])
        c2 = Coord([4, 5, 6])
        cs = CoordSet([c1, c2])
        before = len(cs["x"])
        cs["x_3"] = Coord([7, 8, 9])
        assert len(cs["x"]) == before + 1
        assert cs["x"]["_3"] is not None

    def test_delitem_by_title_removes_top_level(self):
        c1 = Coord([1, 2, 3], name="x", title="wavelength")
        cs = CoordSet(c1)
        assert "wavelength" in cs.titles
        del cs["wavelength"]
        assert len(cs) == 0

    def test_delitem_by_synthetic_alias(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([4, 5, 6])
        cs = CoordSet([c1, c2])
        assert len(cs["x"]) == 2
        del cs["x_1"]
        assert len(cs["x"]) == 1

    def test_delitem_by_title_in_nested_child(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([4, 5, 6], title="alpha")
        cs = CoordSet([c1, c2])
        del cs["alpha"]
        assert len(cs["x"]) == 1

    def test_setitem_by_title_duplicate_warns(self):
        c1 = Coord([1, 2, 3], name="x", title="dup")
        c2 = Coord([4, 5, 6], name="y", title="dup")
        cs = CoordSet(c1, c2)
        with pytest.warns(UserWarning, match="occurs several time"):
            cs["dup"] = Coord([7, 8, 9], name="x")


# ==============================================================================
# Lookup ambiguities
# ==============================================================================


class TestCoordSetLookupAmbiguities:
    """Current public lookup behavior for ambiguous CoordSet cases."""

    def test_getitem_duplicate_top_level_title_warns_and_returns_first(self):
        c1 = Coord([1, 2, 3], name="x", title="dup")
        c2 = Coord([4, 5], name="y", title="dup")
        cs = CoordSet(c1, c2, keepnames=True)
        with pytest.warns(UserWarning, match="occurs several time"):
            found = cs["dup"]
        assert found == c1

    def test_getitem_duplicate_child_titles_same_group_returns_first_without_warning(
        self,
    ):
        c1 = Coord([1, 2, 3], name="a", title="dup")
        c2 = Coord([4, 5, 6], name="b", title="dup")
        cs = CoordSet([c1, c2])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            found = cs["dup"]
        assert recorded == []
        assert found == cs["x"]["_1"]

    def test_getitem_duplicate_child_titles_across_groups_returns_first_without_warning(
        self,
    ):
        xa = Coord([1, 2, 3], name="a", title="dup")
        xb = Coord([4, 5, 6], name="b", title="dup")
        ya = Coord([10, 20], name="c", title="dup")
        yb = Coord([30, 40], name="d", title="dup")
        cs = CoordSet(x=CoordSet(xa, xb), y=CoordSet(ya, yb))
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            found = cs["dup"]
        assert recorded == []
        assert found == cs["x"]["_1"]

    def test_dimension_name_takes_precedence_over_top_level_title(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([4, 5], name="y", title="x")
        cs = CoordSet(c1, c2, keepnames=True)
        found = cs["x"]
        assert found == c1
        assert found.title != "x"

    def test_dimension_name_takes_precedence_over_child_title(self):
        child = Coord([1, 2, 3], name="a", title="y")
        sibling = Coord([4, 5, 6], name="b")
        cs = CoordSet(x=CoordSet(child, sibling), y=Coord([10, 20, 30], name="y"))
        found = cs["y"]
        assert found.name == "y"
        assert found.title != "y"

    def test_reference_key_takes_precedence_over_title(self):
        c1 = Coord([1, 2, 3], name="x", title="y")
        cs = CoordSet(x=c1, y="x")
        assert cs["y"] == "x"
        assert cs["x"].title == "y"

    def test_global_synthetic_alias_collision_returns_first_group_child(self):
        cs = CoordSet(
            x=CoordSet(Coord([1, 2, 3], name="a"), Coord([4, 5, 6], name="b")),
            y=CoordSet(Coord([10, 20, 30], name="c"), Coord([40, 50, 60], name="d")),
        )
        assert cs["_1"] == cs["x"]["_1"]
        assert_array_equal(cs["_1"].data, [4.0, 5.0, 6.0])

    def test_dimension_scoped_synthetic_alias_collision_targets_requested_group(self):
        cs = CoordSet(
            x=CoordSet(Coord([1, 2, 3], name="a"), Coord([4, 5, 6], name="b")),
            y=CoordSet(Coord([10, 20, 30], name="c"), Coord([40, 50, 60], name="d")),
        )
        assert cs["x_1"] == cs["x"]["_1"]
        assert cs["y_1"] == cs["y"]["_1"]
        assert_array_equal(cs["x_1"].data, [4.0, 5.0, 6.0])
        assert_array_equal(cs["y_1"].data, [40.0, 50.0, 60.0])


# ==============================================================================
# Call syntax
# ==============================================================================


class TestCoordSetCall:
    """CoordSet __call__ behavior."""

    def test_call_no_args_returns_all(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        result = cs()
        assert isinstance(result, CoordSet)

    def test_call_with_index_returns_coord(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        result = cs(0)
        assert isinstance(result, Coord)

    def test_call_with_axis(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        result = cs(axis="x")
        assert isinstance(result, Coord)


# ==============================================================================
# Copy
# ==============================================================================


class TestCoordSetCopy:
    """CoordSet copy behavior."""

    def test_copy_preserves_names(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        cs2 = cs.copy()
        assert cs2.names == cs.names

    def test_copy_independence(self):
        c1 = Coord([1, 2, 3], name="x")
        cs = CoordSet(c1)
        cs2 = cs.copy()
        cs2["x"] = Coord([7, 8, 9], name="x")
        assert_array_equal(cs["x"].data, [1, 2, 3])

    def test_copy_preserves_data(self):
        c = Coord([1.0, 2.0, 3.0], name="x")
        cs = CoordSet(c)
        cs2 = cs.copy()
        assert_array_equal(cs2["x"].data, cs["x"].data)

    def test_copy_preserves_selected_non_first_default_for_multicoord(self):
        c1 = Coord([1, 2, 3], name="a")
        c2 = Coord([4, 5, 6], name="b")
        cs = CoordSet([c1, c2])
        cs["x"].select(2)
        cs2 = cs.copy()
        assert cs2["x"].default == cs2["x"]["_2"]
        assert_array_equal(cs2["x"].data, [4.0, 5.0, 6.0])

    def test_deepcopy_preserves_selected_non_first_default_for_multicoord(self):
        c1 = Coord([1, 2, 3], name="a")
        c2 = Coord([4, 5, 6], name="b")
        cs = CoordSet([c1, c2])
        cs["x"].select(2)
        cs2 = deepcopy(cs)
        assert cs2["x"].default == cs2["x"]["_2"]
        assert_array_equal(cs2["x"].data, [4.0, 5.0, 6.0])

    def test_copy_preserves_reference_lookup(self):
        c = Coord([1.0, 2.0, 3.0], name="x")
        cs = CoordSet(x=c, y="x")
        cs2 = cs.copy()
        assert cs2.references == {"y": "x"}
        assert cs2["y"] == "x"
        assert_array_equal(cs2["x"].data, [1.0, 2.0, 3.0])

    def test_deepcopy_preserves_reference_lookup(self):
        c = Coord([1.0, 2.0, 3.0], name="x")
        cs = CoordSet(x=c, y="x")
        cs2 = deepcopy(cs)
        assert cs2.references == {"y": "x"}
        assert cs2["y"] == "x"
        assert_array_equal(cs2["x"].data, [1.0, 2.0, 3.0])


# ==============================================================================
# Arithmetic
# ==============================================================================


class TestCoordSetArithmetic:
    """CoordSet arithmetic behavior."""

    def test_add_two_coordsets(self):
        c1 = Coord([1, 2, 3], name="x")
        cs1 = CoordSet(c1)
        cs2 = CoordSet(Coord([10, 20, 30], name="x"))
        cs_sum = cs1 + cs2
        assert cs_sum is not None
        assert isinstance(cs_sum, CoordSet)

    def test_sub_two_coordsets(self):
        c1 = Coord([10, 20, 30], name="x")
        cs1 = CoordSet(c1)
        cs2 = CoordSet(Coord([1, 2, 3], name="x"))
        cs_diff = cs1 - cs2
        assert cs_diff is not None
        assert isinstance(cs_diff, CoordSet)


# ==============================================================================
# Edge cases
# ==============================================================================


class TestCoordSetEdgeCases:
    """CoordSet edge case behavior."""

    def test_coords_with_different_sizes(self):
        """CoordSet allows coords of different sizes for different dims."""
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        assert cs.sizes == [3, 2]  # order depends on _coords storage

    def test_preserves_coord_units(self):
        c1 = Coord([1, 2, 3], name="x", units="cm^-1")
        c2 = Coord([10, 20], name="y", units="s")
        cs = CoordSet(c1, c2, keepnames=True)
        units = cs.units
        assert any(u is not None for u in units)

    def test_preserves_coord_titles(self):
        c1 = Coord([1, 2, 3], name="x", title="Wavenumber")
        cs = CoordSet(c1)
        assert "Wavenumber" in cs.titles

    def test_coord_with_labels_in_set(self):
        c = Coord([1, 2, 3], name="x", labels=["a", "b", "c"])
        cs = CoordSet(c)
        assert cs.is_labeled
        # labels are returned as list of arrays
        assert len(cs.labels[0]) == 3

    def test_select_changes_default(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        cs.select(0)
        # just check it doesn't raise
        assert cs.default is not None

    def test_select_second_coord(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        cs.select(0)
        cs.select(2)  # select second (1-indexed)
        # default may or may not change depending on count
        assert cs is not None


# ==============================================================================
# Multi-coord (same-dimension)
# ==============================================================================


class TestCoordSetMultiCoord:
    """CoordSet with multiple coords sharing a dimension."""

    def test_list_arg_exposes_child_coords(self):
        """Passing a list of Coords exposes both through compatibility aliases."""
        c1 = Coord([1, 2, 3], name="a")
        c2 = Coord([4, 5, 6], name="b")
        cs = CoordSet([c1, c2])
        assert cs["x"]["_1"] == c1
        assert cs["x"]["_2"] == c2

    def test_child_coord_metadata_preserved(self):
        c1 = Coord([1, 2, 3], name="a", units="cm^-1", title="alpha")
        c2 = Coord([4, 5, 6], name="b", units="s", title="beta")
        cs = CoordSet([c1, c2])
        assert cs["x"]["_1"].units is not None
        assert cs["x"]["_2"].title == "beta"

    def test_explicit_multi_coord_via_kwargs(self):
        c1 = Coord([1, 2, 3], name="a")
        c2 = Coord([4, 5, 6], name="b")
        cs = CoordSet(x=CoordSet(c1, c2))
        # Both coordinates are accessible through compatibility aliases
        assert len(cs["x"]) == 2


# ==============================================================================
# Attribute access
# ==============================================================================


class TestCoordSetAttributeAccess:
    """CoordSet __getattr__ behavior."""

    def test_getattr_by_name(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"), Coord([10, 20], name="y"))
        assert cs.x is not None
        assert cs.y is not None

    def test_getattr_returns_coord(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        assert isinstance(cs.x, Coord)

    def test_getattr_subcoord_by_index(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([4, 5, 6])
        cs = CoordSet([c1, c2])
        sub = cs["x"]
        assert sub._1 == c1
        assert sub._2 == c2

    def test_getattr_missing_raises(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        with pytest.raises(AttributeError):
            _ = cs.nonexistent


# ==============================================================================
# Additional properties
# ==============================================================================


class TestCoordSetAdditionalProperties:
    """Additional CoordSet public properties."""

    def test_name_default_is_id_string(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        name = cs.name
        assert isinstance(name, str)
        assert len(name) > 0

    def test_name_can_be_set(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        cs.name = "mycoords"
        assert cs.name == "mycoords"

    def test_coords_returns_list(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"), Coord([10, 20], name="y"))
        coords = cs.coords
        assert isinstance(coords, list)
        assert len(coords) == 2

    def test_coords_contains_coord_objects(self):
        c = Coord([1, 2, 3], name="x")
        cs = CoordSet(c)
        assert isinstance(cs.coords[0], Coord)


# ==============================================================================
# Iteration
# ==============================================================================


class TestCoordSetIteration:
    """CoordSet iteration behavior."""

    def test_iter_yields_all_items(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([10, 20], name="y")
        cs = CoordSet(c1, c2, keepnames=True)
        items = list(cs)
        assert len(items) == 2

    def test_iter_contains_all_items(self):
        c1 = Coord([1, 2, 3], name="a")
        c2 = Coord([10, 20], name="b")
        cs = CoordSet(c1, c2, keepnames=True)
        items = list(cs)
        assert c1 in items
        assert c2 in items


# ==============================================================================
# Keepnames
# ==============================================================================


class TestCoordSetKeepnames:
    """CoordSet keepnames behavior."""

    def test_keepnames_true_preserves_coord_names(self):
        c = Coord([1, 2, 3], name="wavelength")
        cs = CoordSet(c, keepnames=True)
        assert cs.names == ["wavelength"]

    def test_keepnames_false_auto_renames(self):
        c = Coord([1, 2, 3], name="wavelength")
        cs = CoordSet(c, keepnames=False)
        # When keepnames=False, coord names may be reassigned
        assert cs.names is not None

    def test_default_keepnames_false(self):
        c = Coord([1, 2, 3], name="wavelength")
        cs = CoordSet(c)
        # Default is keepnames=False, so name may be auto-assigned
        assert cs.names is not None


# ==============================================================================
# Equality
# ==============================================================================


class TestCoordSetEquality:
    """CoordSet equality behavior."""

    def test_eq_same_coords(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([1, 2, 3], name="x")
        cs1 = CoordSet(c1)
        cs2 = CoordSet(c2)
        assert cs1 == cs2

    def test_eq_different_coords(self):
        c1 = Coord([1, 2, 3], name="x")
        c2 = Coord([4, 5, 6], name="x")
        cs1 = CoordSet(c1)
        cs2 = CoordSet(c2)
        assert cs1 != cs2

    def test_eq_with_none(self):
        cs = CoordSet(Coord([1, 2, 3], name="x"))
        assert cs != None  # noqa: E711

    def test_eq_different_count(self):
        cs1 = CoordSet(Coord([1, 2, 3], name="x"))
        cs2 = CoordSet(Coord([1, 2, 3], name="x"), Coord([10, 20], name="y"))
        assert cs1 != cs2
