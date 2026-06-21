"""
Characterization tests for shape operation semantics on NDDataset.

This suite characterizes CURRENT behavior of shape-changing operations.
It does NOT validate a desired future policy.

Coverage:
    - transpose
    - swapdims
    - squeeze
    - reshape
    - atleast_2d
    - CoordSet semantics (Preserve / Reduce / Rebuild / Synthesize)
    - history behavior
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_basic_metadata_preserved,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import assert_dims_equal
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_distinct_object,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_history_appended,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_history_preserved,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    make_semantic_2d_dataset,
)

# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def shape_dataset():
    """
    Semantic-rich 2D dataset for shape operation tests.

    - dims: ['y', 'x'] (5, 7)
    - CoordSet with titles, units
    - title, name, metadata, history
    """
    return make_semantic_2d_dataset(
        title="shape_dataset",
        name="shape_name",
        description="test description",
        history=["original entry"],
    )


@pytest.fixture
def shape_dataset_1d():
    """1D dataset with a single coordinate."""
    c = Coord(np.linspace(0.0, 100.0, 10), title="position", units="m")
    ds = NDDataset(
        np.arange(10.0, dtype="float64"), coordset=[c], title="oned", name="one_d"
    )
    ds.history = ["original"]
    return ds


@pytest.fixture
def singleton_dataset():
    """2D dataset with a singleton first dimension."""
    ds = NDDataset(
        np.ones((1, 4)),
        coordset=[None, Coord(np.arange(4.0), title="xval")],
        title="singleton",
    )
    ds.history = ["original"]
    return ds


@pytest.fixture
def multicoord_dataset():
    """3D dataset with a multicoord dimension and a singleton."""
    ds = NDDataset(np.ones((2, 1, 2)), dims=["y", "z", "x"])
    cs_y = CoordSet(
        [
            Coord(np.array([10.0, 20.0]), title="a"),
            Coord(np.array([100.0, 200.0]), title="b"),
        ],
        dim="y",
    )
    ds.y = cs_y
    ds.x = Coord(np.array([1.0, 2.0]), title="c")
    return ds


@pytest.fixture
def bare_dataset():
    """Minimal dataset with no coordset."""
    return NDDataset(np.array([1.0, 2.0, 3.0]))


@pytest.fixture
def zero_dim_dataset():
    """0-dimensional scalar dataset."""
    return NDDataset(42.0)


# ======================================================================================
# TRANSPOSE
# ======================================================================================


class TestTransposeCharacterization:
    """Characterize NDDataset.transpose() current behavior."""

    def test_transpose_reverses_dims(self, shape_dataset):
        t = shape_dataset.transpose()
        assert_dims_equal(t, ["x", "y"])
        assert t.shape == (7, 5)

    def test_transpose_explicit_dims(self, shape_dataset):
        t = shape_dataset.transpose("y", "x")
        assert_dims_equal(t, ["y", "x"])
        assert t.shape == (5, 7)

    def test_transpose_dim_type_is_numpy_string(self, shape_dataset):
        """SURPRISE: transpose dims are numpy.str_, not plain str."""
        t = shape_dataset.transpose()
        for d in t.dims:
            assert isinstance(d, np.str_)

    def test_transpose_preserves_coordset(self, shape_dataset):
        t = shape_dataset.transpose()
        assert t.coordset is not None
        assert t.coordset.names == ["x", "y"]

    def test_transpose_coord_values_unchanged(self, shape_dataset):
        t = shape_dataset.transpose()
        assert np.array_equal(t["y"].data, shape_dataset["y"].data)
        assert np.array_equal(t["x"].data, shape_dataset["x"].data)

    def test_transpose_coord_titles_unchanged(self, shape_dataset):
        t = shape_dataset.transpose()
        assert t["y"].title == "time"
        assert t["x"].title == "wavenumber"

    def test_transpose_coord_units_unchanged(self, shape_dataset):
        t = shape_dataset.transpose()
        assert str(t["y"].units) == "s"
        assert "cm" in str(t["x"].units)

    def test_transpose_preserves_title(self, shape_dataset):
        t = shape_dataset.transpose()
        assert t.title == "shape_dataset"

    def test_transpose_preserves_name(self, shape_dataset):
        t = shape_dataset.transpose()
        assert t.name == "shape_name"

    def test_transpose_preserves_metadata(self, shape_dataset):
        t = shape_dataset.transpose()
        assert_basic_metadata_preserved(t, shape_dataset)

    def test_transpose_appends_history(self, shape_dataset):
        assert_history_appended(
            shape_dataset.transpose(), shape_dataset, "Data transposed"
        )

    def test_transpose_appends_history_with_dims(self, shape_dataset):
        t = shape_dataset.transpose("y", "x")
        assert "Data transposed between dims:" in t.history[-1]
        assert "y" in t.history[-1] and "x" in t.history[-1]

    def test_transpose_returns_new_object(self, shape_dataset):
        assert_distinct_object(shape_dataset.transpose(), shape_dataset)

    def test_transpose_noop_for_1d(self, shape_dataset_1d):
        t = shape_dataset_1d.transpose()
        assert_dims_equal(t, shape_dataset_1d.dims)
        assert t.shape == shape_dataset_1d.shape

    def test_T_property_equals_transpose(self, shape_dataset):
        t1 = shape_dataset.transpose()
        t2 = shape_dataset.T
        assert t1.dims == t2.dims
        assert t1.shape == t2.shape

    def test_transpose_semantic_pattern(self, shape_dataset):
        """CoordSet semantics: Preserve (dim names stay, only axis order changes)."""
        t = shape_dataset.transpose()
        assert t.coordset is not None
        assert set(t.coordset.names) == {"x", "y"}


# ======================================================================================
# SWAPDIMS
# ======================================================================================


class TestSwapdimsCharacterization:
    """Characterize NDDataset.swapdims() current behavior."""

    def test_swapdims_swaps_two_dims(self, shape_dataset):
        s = shape_dataset.swapdims("y", "x")
        assert_dims_equal(s, ["x", "y"])
        assert s.shape == (7, 5)

    def test_swapdims_dim_type_is_plain_string(self, shape_dataset):
        """SURPRISE: swapdims dims are plain str, unlike transpose which gives numpy.str_."""
        s = shape_dataset.swapdims("y", "x")
        for d in s.dims:
            assert isinstance(d, str)

    def test_swapdims_preserves_coordset(self, shape_dataset):
        s = shape_dataset.swapdims("y", "x")
        assert s.coordset is not None
        assert s.coordset.names == ["x", "y"]

    def test_swapdims_coord_values_unchanged(self, shape_dataset):
        s = shape_dataset.swapdims("y", "x")
        assert np.array_equal(s["y"].data, shape_dataset["y"].data)
        assert np.array_equal(s["x"].data, shape_dataset["x"].data)

    def test_swapdims_preserves_title(self, shape_dataset):
        s = shape_dataset.swapdims("y", "x")
        assert s.title == "shape_dataset"

    def test_swapdims_preserves_name(self, shape_dataset):
        s = shape_dataset.swapdims("y", "x")
        assert s.name == "shape_name"

    def test_swapdims_preserves_metadata(self, shape_dataset):
        s = shape_dataset.swapdims("y", "x")
        assert_basic_metadata_preserved(s, shape_dataset)

    def test_swapdims_appends_history(self, shape_dataset):
        assert_history_appended(
            shape_dataset.swapdims("y", "x"),
            shape_dataset,
            "Data swapped between dims y and x",
        )

    def test_swapdims_returns_new_object(self, shape_dataset):
        assert_distinct_object(shape_dataset.swapdims("y", "x"), shape_dataset)

    def test_swapdims_inplace_returns_same(self, shape_dataset):
        s = shape_dataset.swapdims("y", "x", inplace=True)
        assert s is shape_dataset

    def test_swapdims_noop_on_1d(self, shape_dataset_1d):
        """SURPRISE: swapdims on 1D returns a copy without error (no-op)."""
        s = shape_dataset_1d.swapdims("x", "y")
        assert_dims_equal(s, shape_dataset_1d.dims)
        assert s.shape == shape_dataset_1d.shape
        assert_distinct_object(s, shape_dataset_1d)

    def test_swapdims_semantic_pattern(self, shape_dataset):
        """CoordSet semantics: Preserve (dim names swap, coords follow by name)."""
        s = shape_dataset.swapdims("y", "x")
        assert s.coordset is not None
        assert set(s.coordset.names) == {"x", "y"}


# ======================================================================================
# SQUEEZE
# ======================================================================================


class TestSqueezeCharacterization:
    """Characterize NDDataset.squeeze() current behavior."""

    def test_squeeze_noop_when_no_singleton(self, shape_dataset):
        sq = shape_dataset.squeeze()
        assert_dims_equal(sq, shape_dataset.dims)
        assert sq.shape == shape_dataset.shape
        assert sq.coordset is not None

    def test_squeeze_removes_singleton_dim(self, singleton_dataset):
        sq = singleton_dataset.squeeze()
        assert_dims_equal(sq, ["x"])
        assert sq.shape == (4,)

    def test_squeeze_removes_singleton_coord(self, singleton_dataset):
        sq = singleton_dataset.squeeze()
        assert sq.coordset is not None
        assert sq.coordset.names == ["x"]

    def test_squeeze_preserves_nonsingleton_coord(self, singleton_dataset):
        sq = singleton_dataset.squeeze()
        assert np.array_equal(sq["x"].data, np.arange(4.0))

    def test_squeeze_preserves_coord_title(self, singleton_dataset):
        sq = singleton_dataset.squeeze()
        assert sq["x"].title == "xval"

    def test_squeeze_preserves_title(self, singleton_dataset):
        sq = singleton_dataset.squeeze()
        assert sq.title == "singleton"

    def test_squeeze_multiple_singletons(self):
        ds = NDDataset(
            np.ones((1, 4, 1)),
            coordset=[None, Coord(np.arange(4.0), title="vals"), None],
        )
        sq = ds.squeeze()
        assert sq.shape == (4,)
        assert sq.coordset is not None
        assert "vals" in str(sq.coordset)

    def test_squeeze_preserves_multicoord(self, multicoord_dataset):
        sq = multicoord_dataset.squeeze()
        assert_dims_equal(sq, ["y", "x"])
        assert sq.shape == (2, 2)
        assert sq.coordset is not None
        assert sq.coordset["y"].is_same_dim

    def test_squeeze_records_history(self, singleton_dataset):
        """Squeeze now appends a history entry (consistent with other shape ops)."""
        assert_history_appended(
            singleton_dataset.squeeze(), singleton_dataset, "Data squeezed"
        )

    def test_squeeze_returns_new_object(self, singleton_dataset):
        assert_distinct_object(singleton_dataset.squeeze(), singleton_dataset)

    def test_squeeze_inplace_returns_same(self):
        ds = NDDataset(np.ones((1, 4)))
        sq = ds.squeeze(inplace=True)
        assert sq is ds

    def test_squeeze_preserves_metadata(self):
        ds = NDDataset(np.ones((1, 4)), title="squeeze_meta")
        ds.author = "author_x"
        ds.meta.key = "value"
        sq = ds.squeeze()
        assert sq.author == "author_x"
        assert sq.meta.key == "value"

    def test_squeeze_semantic_pattern(self):
        """CoordSet semantics: Reduce (singleton dims and their coords are dropped)."""
        ds = NDDataset(
            np.ones((1, 4)), coordset=[None, Coord(np.arange(4.0), title="xval")]
        )
        sq = ds.squeeze()
        assert sq.coordset is not None
        assert sq.coordset.names == ["x"]


# ======================================================================================
# RESHAPE
# ======================================================================================


class TestReshapeCharacterization:
    """Characterize NDDataset.reshape() current behavior."""

    def test_reshape_noop_preserves_everything(self, shape_dataset):
        r = shape_dataset.reshape((5, 7))
        assert_dims_equal(r, shape_dataset.dims)
        assert r.shape == (5, 7)
        assert r.coordset is not None
        assert r.title == "shape_dataset"
        assert r.name == "shape_name"

    def test_reshape_with_explicit_dims(self, shape_dataset):
        r = shape_dataset.reshape((7, 5), dims=("x", "y"))
        assert_dims_equal(r, ["x", "y"])
        assert r.shape == (7, 5)
        assert r.coordset is not None

    def test_reshape_preserves_coord_values(self, shape_dataset):
        r = shape_dataset.reshape((7, 5), dims=("x", "y"))
        assert np.array_equal(r["y"].data, shape_dataset["y"].data)
        assert np.array_equal(r["x"].data, shape_dataset["x"].data)

    def test_reshape_preserves_title(self, shape_dataset):
        r = shape_dataset.reshape((35,))
        assert r.title == "shape_dataset"

    def test_reshape_preserves_name(self, shape_dataset):
        r = shape_dataset.reshape((35,))
        assert r.name == "shape_name"

    def test_reshape_preserves_metadata(self, shape_dataset):
        r = shape_dataset.reshape((35,))
        assert_basic_metadata_preserved(r, shape_dataset)

    def test_reshape_appends_history(self, shape_dataset):
        assert_history_appended(
            shape_dataset.reshape((7, 5), dims=("x", "y")),
            shape_dataset,
            "Data reshaped from",
        )

    def test_reshape_drop_policy_clears_coordset(self, shape_dataset):
        r = shape_dataset.reshape((35,), coord_policy="drop")
        assert r.coordset is None

    def test_reshape_strict_preserves_unambiguous(self, shape_dataset):
        r = shape_dataset.reshape((7, 5), dims=("x", "y"), coord_policy="strict")
        assert r.coordset is not None
        assert r.coordset.names == ["x", "y"]

    def test_reshape_safe_drops_ambiguous_coords(self, shape_dataset):
        """SURPRISE: safe policy drops coords when dim sizes change ambiguously."""
        r = shape_dataset.reshape((35,))
        assert r.coordset is None

    def test_reshape_returns_new_object(self, shape_dataset):
        assert_distinct_object(
            shape_dataset.reshape((7, 5), dims=("x", "y")), shape_dataset
        )

    def test_reshape_with_explicit_coords(self, shape_dataset):
        c = Coord(np.linspace(0.0, 60.0, 7), title="new_time", units="s")
        r = shape_dataset.reshape((7, 5), dims=("y", "x"), coords={"y": c})
        assert r.coordset is not None
        assert np.array_equal(r["y"].data, c.data)

    def test_reshape_semantic_pattern(self):
        """CoordSet semantics: Rebuild (coords may be preserved, dropped, or replaced)."""
        ds = NDDataset(
            np.arange(35.0).reshape(5, 7),
            coordset=[
                Coord(np.linspace(0, 60, 5), title="time"),
                Coord(np.linspace(4000, 1000, 7), title="wn"),
            ],
        )
        r = ds.reshape((7, 5), dims=("x", "y"))
        assert r.coordset is not None
        r_drop = ds.reshape((35,), coord_policy="drop")
        assert r_drop.coordset is None


# ======================================================================================
# ATLEAST_2D
# ======================================================================================


class TestAtleast2dCharacterization:
    """Characterize NDDataset.atleast_2d() current behavior."""

    def test_atleast_2d_1d_to_2d(self, shape_dataset_1d):
        a = shape_dataset_1d.atleast_2d()
        assert a.ndim == 2
        assert a.shape == (1, 10)
        assert_dims_equal(a, ["u", "x"])

    def test_atleast_2d_1d_preserves_original_coord(self, shape_dataset_1d):
        a = shape_dataset_1d.atleast_2d()
        assert a.coordset is not None
        assert a.coordset.names == ["u", "x"]
        assert np.array_equal(a["x"].data, shape_dataset_1d["x"].data)
        assert a["x"].title == "position"

    def test_atleast_2d_0d_to_2d(self, zero_dim_dataset):
        a = zero_dim_dataset.atleast_2d()
        assert a.ndim == 2
        assert a.shape == (1, 1)
        assert_dims_equal(a, ["v", "u"])
        assert a.coordset is None

    def test_atleast_2d_2d_returns_new_object(self, shape_dataset):
        """SURPRISE: atleast_2d on 2D returns a copy, not self."""
        a = shape_dataset.atleast_2d()
        assert_distinct_object(a, shape_dataset)
        assert a.shape == shape_dataset.shape

    def test_atleast_2d_preserves_title(self, shape_dataset_1d):
        a = shape_dataset_1d.atleast_2d()
        assert a.title == "oned"

    def test_atleast_2d_preserves_name(self, shape_dataset_1d):
        a = shape_dataset_1d.atleast_2d()
        assert a.name == "one_d"

    def test_atleast_2d_does_not_record_history(self, shape_dataset_1d):
        """SURPRISE: atleast_2d does NOT append history."""
        a = shape_dataset_1d.atleast_2d()
        assert len(a.history) == 1
        assert a.history[0] == shape_dataset_1d.history[0]

    def test_atleast_2d_semantic_pattern(self):
        """CoordSet semantics: Rebuild (new dim 'u' gets None coord, original preserved)."""
        ds = NDDataset(
            np.array([1.0, 2.0, 3.0]),
            coordset=[Coord(np.array([10.0, 20.0, 30.0]), title="freq")],
        )
        a = ds.atleast_2d()
        assert a.coordset is not None
        assert a["u"].title == "<untitled>"
        assert a["x"].title == "freq"


# ======================================================================================
# COORDSET SEMANTICS CLASSIFICATION
# ======================================================================================


class TestCoordSetSemanticsClassification:
    """Classify shape operations into Preserve / Reduce / Rebuild / Synthesize patterns."""

    def test_transpose_is_preserve(self, shape_dataset):
        """Transpose preserves all coordinates; only axis order changes."""
        t = shape_dataset.transpose()
        assert t.coordset is not None
        assert len(t.coordset) == len(shape_dataset.coordset)
        for name in shape_dataset.coordset.names:
            assert np.array_equal(t[name].data, shape_dataset[name].data)

    def test_swapdims_is_preserve(self, shape_dataset):
        """Swapdims preserves all coordinates; two dim names are swapped."""
        s = shape_dataset.swapdims("y", "x")
        assert s.coordset is not None
        assert len(s.coordset) == len(shape_dataset.coordset)
        for name in shape_dataset.coordset.names:
            assert np.array_equal(s[name].data, shape_dataset[name].data)

    def test_squeeze_is_reduce(self, singleton_dataset):
        """Squeeze removes singleton dims and their coordinates."""
        sq = singleton_dataset.squeeze()
        assert sq.coordset is not None
        assert len(sq.coordset) < len(singleton_dataset.coordset)
        assert "y" not in sq.coordset.names

    def test_squeeze_noop_is_noop(self, shape_dataset):
        """Squeeze on a dataset with no singleton dims is a no-op."""
        sq = shape_dataset.squeeze()
        assert sq.dims == shape_dataset.dims
        assert sq.coordset.names == shape_dataset.coordset.names

    def test_reshape_coord_policy_safe_is_rebuild(self, shape_dataset):
        """Reshape with safe policy may drop coords when dimensions change."""
        r = shape_dataset.reshape((35,))
        assert r.coordset is None

    def test_reshape_coord_policy_drop_is_synthesize(self, shape_dataset):
        """Reshape with drop policy explicitly synthesizes a new coordset (None)."""
        r = shape_dataset.reshape((35,), coord_policy="drop")
        assert r.coordset is None

    def test_reshape_strict_is_preserve_or_error(self, shape_dataset):
        """Reshape with strict policy preserves coords if unambiguous."""
        r = shape_dataset.reshape((7, 5), dims=("x", "y"), coord_policy="strict")
        assert r.coordset is not None
        assert "x" in r.coordset.names

    def test_atleast_2d_1d_is_rebuild(self):
        """atleast_2d on 1D rebuilds coordset: adds new dim 'u' with None coord."""
        ds = NDDataset(
            np.array([1.0, 2.0, 3.0]),
            coordset=[Coord(np.array([10.0, 20.0, 30.0]), title="freq")],
        )
        a = ds.atleast_2d()
        assert a.coordset is not None
        assert "u" in a.coordset.names
        assert "x" in a.coordset.names

    def test_atleast_2d_0d_is_synthesize(self, zero_dim_dataset):
        """atleast_2d on 0D synthesizes a new coordset (None)."""
        a = zero_dim_dataset.atleast_2d()
        assert a.coordset is None

    def test_atleast_2d_2d_is_preserve_or_noop(self, shape_dataset):
        """atleast_2d on 2D preserves the existing coordset."""
        a = shape_dataset.atleast_2d()
        assert a.coordset is not None
        assert a.coordset.names == shape_dataset.coordset.names


# ======================================================================================
# HISTORY BEHAVIOR FOR SHAPE OPERATIONS
# ======================================================================================


class TestShapeOperationHistory:
    """Characterize history behavior for each shape operation."""

    def test_transpose_appends_history(self, shape_dataset):
        t = shape_dataset.transpose()
        assert_history_appended(t, shape_dataset, "Data transposed")
        assert isinstance(t.history[-1], str)

    def test_swapdims_appends_history(self, shape_dataset):
        s = shape_dataset.swapdims("y", "x")
        assert_history_appended(s, shape_dataset, "Data swapped")
        assert isinstance(s.history[-1], str)

    def test_squeeze_appends_history(self, singleton_dataset):
        """Squeeze now appends history (consistent with other shape ops)."""
        assert_history_appended(
            singleton_dataset.squeeze(), singleton_dataset, "Data squeezed"
        )

    def test_reshape_appends_history(self, shape_dataset):
        assert_history_appended(
            shape_dataset.reshape((7, 5), dims=("x", "y")),
            shape_dataset,
            "Data reshaped",
        )

    def test_atleast_2d_does_not_append_history(self, shape_dataset_1d):
        """SURPRISE: atleast_2d does NOT record history."""
        assert_history_preserved(shape_dataset_1d.atleast_2d(), shape_dataset_1d)

    def test_history_preserves_original_entry(self, shape_dataset):
        """Original history entry is preserved, not replaced."""
        t = shape_dataset.transpose()
        assert "original entry" in t.history[0].lower()

    def test_history_is_list_of_strings(self, shape_dataset):
        t = shape_dataset.transpose()
        for entry in t.history:
            assert isinstance(entry, str)


# ======================================================================================
# ADDITIONAL SHAPE OPERATIONS
# ======================================================================================


class TestExpandDims:
    """Check expand_dims availability."""

    def test_expand_dims_not_available_as_method(self):
        """expand_dims is NOT available as a method on NDDataset."""
        ds = NDDataset(np.array([1.0, 2.0, 3.0]))
        assert not hasattr(ds, "expand_dims")
        assert "expand_dims" not in dir(NDDataset)


class TestBareDatasetShapeBehavior:
    """Shape operations on bare (no CoordSet) datasets."""

    def test_transpose_bare(self, bare_dataset):
        t = bare_dataset.transpose()
        assert t.dims == bare_dataset.dims
        assert t.shape == bare_dataset.shape

    def test_squeeze_bare(self):
        ds = NDDataset(np.ones((1, 4)))
        sq = ds.squeeze()
        assert sq.shape == (4,)
        assert sq.coordset is None

    def test_reshape_bare(self, bare_dataset):
        r = bare_dataset.reshape((1, 3))
        assert r.shape == (1, 3)
        assert r.coordset is None

    def test_atleast_2d_bare(self, bare_dataset):
        """Bare 1D dataset atleast_2d has no coordset (no source coords to carry)."""
        a = bare_dataset.atleast_2d()
        assert a.shape == (1, 3)
        assert a.coordset is None
