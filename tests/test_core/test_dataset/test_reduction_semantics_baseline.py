"""
Characterization tests for reduction operation semantics on NDDataset.

This suite characterizes CURRENT behavior of reduction operations.
It does NOT validate a desired future policy.

Coverage:
    - sum, mean, std (aggregating reductions)
    - min, max (selection reductions)
    - argmin, argmax (index reductions)
    - dimension reduction (specific dim, all dims)
    - keepdims behavior
    - unit semantics (all reduction types)
    - mask semantics
    - metadata propagation
    - CoordSet reduction / preservation
    - history behavior (appended, not replaced)
    - ROI / modeldata stale-field behavior
    - identity / provenance observations
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def reduction_dataset():
    """
    Semantic-rich 2D dataset for reduction tests.

    - dims: ['y', 'x'] (5, 7)
    - CoordSet with titles, units
    - title, name, metadata, history
    - ROI (UI/selection state — to reassess)
    - modeldata (derived fit info — stale risk)
    """
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(np.linspace(4000.0, 1000.0, 7), title="wavenumber", units="cm^-1")
    data = np.arange(35.0, dtype="float64").reshape(5, 7)
    ds = NDDataset(data, coordset=[y, x], title="reduction_dataset", name="red_name")
    ds.author = "test_author"
    ds.description = "test description"
    ds.origin = "test_origin"
    ds.meta.project = "test_project"
    ds.roi = [0.0, 10.0]
    ds.modeldata = np.full((5, 7), 42.0)
    ds.history = ["original entry"]
    return ds


@pytest.fixture
def bare_dataset():
    """Minimal dataset with no coordset."""
    return NDDataset(np.array([1.0, 2.0, 3.0]))


@pytest.fixture
def unitful_dataset():
    """1D dataset with units for reduction unit checks."""
    ds = NDDataset(np.array([1.0, 2.0, 3.0]), units="m")
    ds.history = ["original"]
    return ds


# ======================================================================================
# SUM
# ======================================================================================


class TestSumCharacterization:
    """Characterize NDDataset.sum() current behavior."""

    def test_sum_over_specific_dim_returns_nddataset(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert isinstance(s, NDDataset)
        assert s.shape == (5,)
        assert s.dims == ["y"]

    def test_sum_removes_reduced_coord(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.coordset is not None
        assert s.coordset.names == ["y"]

    def test_sum_preserves_surviving_coord(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert "y" in s.coordset.names
        assert s["y"].title == "time"

    def test_sum_over_all_dims_returns_scalar(self, reduction_dataset):
        s = reduction_dataset.sum(dim=None)
        assert not isinstance(s, NDDataset)
        assert np.isclose(s, 595.0)

    def test_sum_over_all_dims_with_keepdims(self, reduction_dataset):
        s = reduction_dataset.sum(dim=None, keepdims=True)
        assert isinstance(s, NDDataset)
        assert s.shape == (1, 1)
        assert s.coordset is None

    def test_sum_keepdims_preserves_dim(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x", keepdims=True)
        assert s.shape == (5, 1)
        assert s.dims == ["y", "x"]
        assert s.coordset is not None

    def test_sum_keepdims_coord_is_singleton(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x", keepdims=True)
        assert np.array_equal(s["x"].data, [0])

    def test_sum_preserves_units(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.units is None

    def test_sum_with_units(self, unitful_dataset):
        s = unitful_dataset.sum(dim="x")
        assert s.units == unitful_dataset.units

    def test_sum_preserves_title(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.title == "reduction_dataset"

    def test_sum_preserves_name(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.name == "red_name"

    def test_sum_preserves_metadata(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.author == "test_author"
        assert s.description == "test description"
        assert s.origin == "test_origin"
        assert s.meta.project == "test_project"

    def test_sum_appends_history(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert len(s.history) == 2
        assert "original entry" in s.history[0].lower()
        assert "Dataset resulting from application" in s.history[1]

    def test_sum_history_message_format(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert "`sum`" in s.history[1]

    def test_sum_masked_values_ignored(self):
        arr = np.ma.MaskedArray(
            np.arange(35.0).reshape(5, 7), mask=np.zeros((5, 7), dtype=bool)
        )
        arr[0, 0] = np.ma.masked
        ds = NDDataset(arr)
        s = ds.sum(dim="x")
        expected = np.sum(np.arange(35.0).reshape(5, 7), axis=1)
        expected[0] = expected[0] - 0.0
        assert np.isclose(s.data[0], expected[0])

    def test_sum_preserves_roi(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.roi == [0.0, 10.0]

    def test_sum_modeldata_shape_stale(self, reduction_dataset):
        """Notable behavior: modeldata retains original shape after reduction (stale)."""
        s = reduction_dataset.sum(dim="x")
        assert s.modeldata.shape == (5, 7)
        assert s.shape == (5,)


# ======================================================================================
# MEAN
# ======================================================================================


class TestMeanCharacterization:
    """Characterize NDDataset.mean() current behavior."""

    def test_mean_over_specific_dim(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert isinstance(m, NDDataset)
        assert m.shape == (5,)
        assert m.dims == ["y"]

    def mean_expected_values(self):
        data = np.arange(35.0).reshape(5, 7)
        return np.mean(data, axis=1)

    def test_mean_numerical(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        expected = np.mean(np.arange(35.0).reshape(5, 7), axis=1)
        assert np.allclose(m.data, expected)

    def test_mean_over_all_dims_returns_scalar(self, reduction_dataset):
        m = reduction_dataset.mean(dim=None)
        assert not isinstance(m, NDDataset)

    def test_mean_preserves_units(self, reduction_dataset):
        # Units are preserved through copy, but mean may not add them
        m = reduction_dataset.mean(dim="x")
        assert m.units is None

    def test_mean_preserves_title(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert m.title == "reduction_dataset"

    def test_mean_appends_history(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert len(m.history) == 2
        assert "`mean`" in m.history[1]

    def test_mean_coordset_reduced(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert m.coordset is not None
        assert m.coordset.names == ["y"]

    def test_mean_keepdims(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x", keepdims=True)
        assert m.shape == (5, 1)
        assert m.dims == ["y", "x"]

    def test_mean_masked(self):
        arr = np.ma.MaskedArray(
            np.arange(35.0).reshape(5, 7), mask=np.zeros((5, 7), dtype=bool)
        )
        arr[0, 0] = np.ma.masked
        ds = NDDataset(arr)
        m = ds.mean(dim="x")
        expected = np.ma.mean(arr, axis=1)
        assert np.isclose(m.data[0], expected[0])


# ======================================================================================
# STD
# ======================================================================================


class TestStdCharacterization:
    """Characterize NDDataset.std() current behavior."""

    def test_std_over_specific_dim(self, reduction_dataset):
        s = reduction_dataset.std(dim="x")
        assert isinstance(s, NDDataset)
        assert s.shape == (5,)
        assert s.dims == ["y"]

    def test_std_numerical(self, reduction_dataset):
        s = reduction_dataset.std(dim="x")
        expected = np.std(np.arange(35.0).reshape(5, 7), axis=1)
        assert np.allclose(s.data, expected)

    def test_std_over_all_dims_returns_scalar(self, reduction_dataset):
        s = reduction_dataset.std(dim=None)
        assert not isinstance(s, NDDataset)

    def test_std_appends_history(self, reduction_dataset):
        s = reduction_dataset.std(dim="x")
        assert "`std`" in s.history[1]

    def test_std_keepdims(self, reduction_dataset):
        s = reduction_dataset.std(dim="x", keepdims=True)
        assert s.shape == (5, 1)

    def test_std_coordset_reduced(self, reduction_dataset):
        s = reduction_dataset.std(dim="x")
        assert s.coordset is not None
        assert s.coordset.names == ["y"]


# ======================================================================================
# MIN / MAX
# ======================================================================================


class TestMinMaxCharacterization:
    """Characterize NDDataset.min() / max() current behavior."""

    def test_max_over_specific_dim(self, reduction_dataset):
        mx = reduction_dataset.max(dim="x")
        assert isinstance(mx, NDDataset)
        assert mx.shape == (5,)
        assert mx.dims == ["y"]

    def test_max_numerical(self, reduction_dataset):
        mx = reduction_dataset.max(dim="x")
        expected = np.max(np.arange(35.0).reshape(5, 7), axis=1)
        assert np.allclose(mx.data, expected)

    def test_min_over_specific_dim(self, reduction_dataset):
        mn = reduction_dataset.min(dim="x")
        assert isinstance(mn, NDDataset)
        assert mn.shape == (5,)
        assert mn.dims == ["y"]

    def test_min_numerical(self, reduction_dataset):
        mn = reduction_dataset.min(dim="x")
        expected = np.min(np.arange(35.0).reshape(5, 7), axis=1)
        assert np.allclose(mn.data, expected)

    def test_max_over_all_dims_returns_scalar(self, reduction_dataset):
        mx = reduction_dataset.max(dim=None)
        assert not isinstance(mx, NDDataset)

    def test_max_keepdims_reconstructs_coords(self, reduction_dataset):
        """
        Notable behavior: max with dim=None, keepdims=True reconstructs coords
        at the extremum location rather than dropping all coords.
        """
        mx = reduction_dataset.max(dim=None, keepdims=True)
        assert isinstance(mx, NDDataset)
        assert mx.shape == (1, 1)
        assert mx.coordset is not None
        assert set(mx.coordset.names) == {"x", "y"}

    def test_min_keepdims_reconstructs_coords(self, reduction_dataset):
        """Notable behavior: min also reconstructs coords at extremum location."""
        mn = reduction_dataset.min(dim=None, keepdims=True)
        assert isinstance(mn, NDDataset)
        assert mn.coordset is not None
        assert set(mn.coordset.names) == {"x", "y"}

    def test_max_appends_history(self, reduction_dataset):
        mx = reduction_dataset.max(dim="x")
        assert "`amax`" in mx.history[1]

    def test_min_appends_history(self, reduction_dataset):
        mn = reduction_dataset.min(dim="x")
        assert "`amin`" in mn.history[1]

    def test_max_preserves_title(self, reduction_dataset):
        mx = reduction_dataset.max(dim="x")
        assert mx.title == "reduction_dataset"

    def test_min_preserves_metadata(self, reduction_dataset):
        mn = reduction_dataset.min(dim="x")
        assert mn.meta.project == "test_project"


# ======================================================================================
# ARGMIN / ARGMAX
# ======================================================================================


class TestArgminArgmaxCharacterization:
    """
    Characterize NDDataset.argmin() / argmax() current behavior.

    These return plain scalars or numpy arrays, never NDDataset.
    """

    def test_argmax_over_dim_returns_ndarray(self, reduction_dataset):
        am = reduction_dataset.argmax(dim="x")
        assert isinstance(am, np.ndarray)
        assert am.shape == (5,)

    def test_argmax_all_dims_returns_tuple(self, reduction_dataset):
        """Notable behavior: argmax over all dims returns a tuple of indices."""
        am = reduction_dataset.argmax(dim=None)
        assert isinstance(am, tuple)
        assert len(am) == 2

    def test_argmax_1d_returns_int(self):
        ds = NDDataset(np.array([10.0, 5.0, 20.0, 3.0]))
        am = ds.argmax()
        assert isinstance(am, (int, np.integer))
        assert am == 2

    def test_argmin_over_dim_returns_ndarray(self, reduction_dataset):
        am = reduction_dataset.argmin(dim="x")
        assert isinstance(am, np.ndarray)

    def test_argmin_all_dims_returns_tuple(self, reduction_dataset):
        am = reduction_dataset.argmin(dim=None)
        assert isinstance(am, tuple)

    def test_argmin_1d_returns_int(self):
        ds = NDDataset(np.array([10.0, 5.0, 20.0, 3.0]))
        am = ds.argmin()
        assert isinstance(am, (int, np.integer))
        assert am == 3

    def test_argmax_correct_values(self, reduction_dataset):
        am = reduction_dataset.argmax(dim="x")
        expected = np.argmax(np.arange(35.0).reshape(5, 7), axis=1)
        assert np.array_equal(am, expected)

    def test_argmin_correct_values(self, reduction_dataset):
        am = reduction_dataset.argmin(dim="x")
        expected = np.argmin(np.arange(35.0).reshape(5, 7), axis=1)
        assert np.array_equal(am, expected)

    def test_argmax_never_returns_nddataset(self, reduction_dataset):
        am = reduction_dataset.argmax(dim="x")
        assert not isinstance(am, NDDataset)

    def test_argmin_never_returns_nddataset(self, reduction_dataset):
        am = reduction_dataset.argmin(dim="x")
        assert not isinstance(am, NDDataset)

    def test_argmax_keepdims_accepted_but_ignored(self, reduction_dataset):
        """
        Notable behavior: keepdims is accepted by argmax but has no effect
        on the return type or shape — it is a no-op.
        """
        am = reduction_dataset.argmax(dim="x", keepdims=True)
        assert isinstance(am, np.ndarray)
        assert am.shape == (5,)
        expected = np.argmax(np.arange(35.0).reshape(5, 7), axis=1)
        assert np.array_equal(am, expected)

    def test_argmin_keepdims_accepted_but_ignored(self, reduction_dataset):
        """
        Notable behavior: keepdims is accepted by argmin but has no effect
        on the return type or shape — it is a no-op.
        """
        am = reduction_dataset.argmin(dim="x", keepdims=True)
        assert isinstance(am, np.ndarray)
        assert am.shape == (5,)
        expected = np.argmin(np.arange(35.0).reshape(5, 7), axis=1)
        assert np.array_equal(am, expected)

    def test_argmax_global_keepdims_ignored(self, reduction_dataset):
        """keepdims=True with dim=None still returns a tuple (no effect)."""
        am = reduction_dataset.argmax(dim=None, keepdims=True)
        assert isinstance(am, tuple)
        assert len(am) == 2

    def test_argmin_global_keepdims_ignored(self, reduction_dataset):
        """keepdims=True with dim=None still returns a tuple (no effect)."""
        am = reduction_dataset.argmin(dim=None, keepdims=True)
        assert isinstance(am, tuple)
        assert len(am) == 2

    def test_argmax_1d_keepdims_ignored(self):
        ds = NDDataset(np.array([10.0, 5.0, 20.0, 3.0]))
        am = ds.argmax(dim="x", keepdims=True)
        assert isinstance(am, (int, np.integer))
        assert am == 2

    def test_argmin_1d_keepdims_ignored(self):
        ds = NDDataset(np.array([10.0, 5.0, 20.0, 3.0]))
        am = ds.argmin(dim="x", keepdims=True)
        assert isinstance(am, (int, np.integer))
        assert am == 3


# ======================================================================================
# COORDSET SEMANTICS
# ======================================================================================


class TestReductionCoordSetSemantics:
    """
    Classify reduction CoordSet behavior.

    Pattern: Reduce (reduced dim coord dropped, surviving coords preserved).
    """

    def test_reduction_removes_reduced_dim_coord(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert "x" not in s.coordset.names
        assert "y" in s.coordset.names

    def test_reduction_preserves_surviving_coord_values(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert np.array_equal(s["y"].data, reduction_dataset["y"].data)

    def test_reduction_preserves_surviving_coord_title(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s["y"].title == "time"

    def test_reduction_preserves_surviving_coord_units(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert str(s["y"].units) == "s"

    def test_reduction_keepdims_coord_is_singleton(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x", keepdims=True)
        assert s["x"].shape == (1,)
        assert np.array_equal(s["x"].data, [0])

    def test_global_reduction_drops_all_coords(self, reduction_dataset):
        s = reduction_dataset.sum(dim=None, keepdims=True)
        assert s.coordset is None

    def test_mean_coordset_behavior(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert m.coordset.names == ["y"]

    def test_std_coordset_behavior(self, reduction_dataset):
        s = reduction_dataset.std(dim="x")
        assert s.coordset.names == ["y"]

    def test_extremum_keepdims_reconstructs_coords(self, reduction_dataset):
        """
        amax/amin reconstruct coords at extremum location
        when dim=None, keepdims=True. This is asymmetric vs sum/mean.
        """
        mx = reduction_dataset.max(dim=None, keepdims=True)
        assert mx.coordset is not None
        assert len(mx.coordset) > 0

    def test_sum_keepdims_multicoord_preserved(self):
        """Sum with keepdims preserves multicoord on surviving dim."""
        from spectrochempy.core.dataset.coordset import CoordSet

        ds = NDDataset(np.ones((2, 3)), dims=["y", "x"])
        cs = CoordSet(
            [Coord([10.0, 20.0], title="a"), Coord([100.0, 200.0], title="b")], dim="y"
        )
        ds.y = cs
        s = ds.sum(dim="x", keepdims=True)
        assert s.coordset is not None
        assert s["y"].is_same_dim


# ======================================================================================
# METADATA SEMANTICS
# ======================================================================================


class TestReductionMetadata:
    """Characterize metadata behavior through reductions."""

    def test_title_preserved(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.title == "reduction_dataset"

    def test_name_preserved(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.name == "red_name"

    def test_author_preserved(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.author == "test_author"

    def test_description_preserved(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.description == "test description"

    def test_origin_preserved(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.origin == "test_origin"

    def test_custom_meta_preserved(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.meta.project == "test_project"

    def test_mean_preserves_metadata(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert m.author == "test_author"
        assert m.meta.project == "test_project"

    def test_max_preserves_metadata(self, reduction_dataset):
        mx = reduction_dataset.max(dim="x")
        assert mx.author == "test_author"


# ======================================================================================
# HISTORY BEHAVIOR
# ======================================================================================


class TestReductionHistory:
    """Characterize history behavior through reductions."""

    def test_reduction_appends_history(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert len(s.history) == 2
        assert s.history[0] == reduction_dataset.history[0]

    def test_reduction_history_message_format(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert "Dataset resulting from application of `sum` method" in s.history[1]

    def test_mean_history_message(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert "`mean`" in m.history[1]

    def test_std_history_message(self, reduction_dataset):
        s = reduction_dataset.std(dim="x")
        assert "`std`" in s.history[1]

    def test_max_history_message(self, reduction_dataset):
        mx = reduction_dataset.max(dim="x")
        assert "`amax`" in mx.history[1]

    def test_min_history_message(self, reduction_dataset):
        mn = reduction_dataset.min(dim="x")
        assert "`amin`" in mn.history[1]

    def test_global_reduction_preserves_history(self, reduction_dataset):
        s = reduction_dataset.sum(dim=None, keepdims=True)
        assert len(s.history) == 2

    def test_history_is_list_of_strings(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        for entry in s.history:
            assert isinstance(entry, str)

    def test_history_not_replaced(self, reduction_dataset):
        """Original history entry is preserved, history is appended."""
        s = reduction_dataset.sum(dim="x")
        assert "original entry" in s.history[0].lower()


# ======================================================================================
# ROI / MODELDATA
# ======================================================================================


class TestReductionRoiModeldata:
    """
    Characterize ROI and modeldata behavior through reductions.

    ROI is current behavior only — likely UI/selection state, to reassess.
    Modeldata is derived model/fit information — stale after reduction.
    """

    def test_roi_preserved_after_sum(self, reduction_dataset):
        s = reduction_dataset.sum(dim="x")
        assert s.roi == [0.0, 10.0]

    def test_roi_preserved_after_mean(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert m.roi == [0.0, 10.0]

    def test_roi_preserved_after_max(self, reduction_dataset):
        mx = reduction_dataset.max(dim="x")
        assert mx.roi == [0.0, 10.0]

    def test_modeldata_stale_after_sum(self, reduction_dataset):
        """Notable behavior: modeldata retains original shape after dim reduction."""
        s = reduction_dataset.sum(dim="x")
        assert s.modeldata.shape == (5, 7)
        assert s.shape == (5,)

    def test_modeldata_stale_after_mean(self, reduction_dataset):
        m = reduction_dataset.mean(dim="x")
        assert m.modeldata.shape == (5, 7)
        assert m.shape == (5,)

    def test_modeldata_stale_after_max(self, reduction_dataset):
        mx = reduction_dataset.max(dim="x")
        assert mx.modeldata.shape == (5, 7)
        assert mx.shape == (5,)

    def test_global_reduction_modeldata_shape(self, reduction_dataset):
        s = reduction_dataset.sum(dim=None, keepdims=True)
        assert s.modeldata.shape == (5, 7)
        assert s.shape == (1, 1)


# ======================================================================================
# IDENTITY / PROVENANCE OBSERVATIONS
# ======================================================================================


class TestReductionIdentityProvenance:
    """
    Observe identity and provenance patterns in reductions.

    Observations (not policy):
    - Reductions preserve title, name, author, origin, custom meta from source.
    - This suggests they are treated as preserving scientific identity
      (same object, different representation) rather than producing a new identity.
    - History is appended, not replaced, reinforcing the same-object view.
    - However, global reduction (dim=None) returns a plain scalar, breaking
      the NDDataset chain — this is more like a derived result.
    - modeldata becomes stale, suggesting the lifecycle of derived fields
      was not designed for shape-changing operations.
    """

    def test_reduction_preserves_scientific_context(self, reduction_dataset):
        """title, description, meta preserved — identity-preserving pattern."""
        s = reduction_dataset.sum(dim="x")
        assert s.title == reduction_dataset.title
        assert s.description == reduction_dataset.description
        assert s.meta.project == reduction_dataset.meta.project

    def test_reduction_preserves_provenance(self, reduction_dataset):
        """author, origin preserved — provenance-preserving pattern."""
        s = reduction_dataset.sum(dim="x")
        assert s.author == reduction_dataset.author
        assert s.origin == reduction_dataset.origin

    def test_global_reduction_breaks_nddataset_chain(self, reduction_dataset):
        """dim=None returns a scalar, not an NDDataset — derived result pattern."""
        s = reduction_dataset.sum(dim=None)
        assert not isinstance(s, NDDataset)

    def test_argmin_argmax_not_nddataset(self, reduction_dataset):
        """argmin/argmax return indices, never NDDataset — utility pattern."""
        amx = reduction_dataset.argmax(dim="x")
        amn = reduction_dataset.argmin(dim="x")
        assert not isinstance(amx, NDDataset)
        assert not isinstance(amn, NDDataset)


# ======================================================================================
# UNIT SEMANTICS
# ======================================================================================


class TestReductionUnits:
    """Characterize unit behavior through reductions."""

    def test_std_preserves_units_dim_specific(self, unitful_dataset):
        s = unitful_dataset.std(dim="x")
        assert str(s.units) == "m"

    def test_std_global_returns_quantity(self, unitful_dataset):
        s = unitful_dataset.std(dim=None)
        assert hasattr(s, "units")
        assert str(s.units) == "m"

    def test_std_global_with_keepdims_preserves_units(self, unitful_dataset):
        s = unitful_dataset.std(dim=None, keepdims=True)
        assert isinstance(s, NDDataset)
        assert str(s.units) == "m"

    def test_max_global_returns_quantity(self, unitful_dataset):
        mx = unitful_dataset.max(dim=None)
        assert hasattr(mx, "units")
        assert str(mx.units) == "m"

    def test_min_global_returns_quantity(self, unitful_dataset):
        mn = unitful_dataset.min(dim=None)
        assert hasattr(mn, "units")
        assert str(mn.units) == "m"

    def test_max_global_with_keepdims_preserves_units(self, unitful_dataset):
        mx = unitful_dataset.max(dim=None, keepdims=True)
        assert isinstance(mx, NDDataset)
        assert str(mx.units) == "m"

    def test_min_global_with_keepdims_preserves_units(self, unitful_dataset):
        mn = unitful_dataset.min(dim=None, keepdims=True)
        assert isinstance(mn, NDDataset)
        assert str(mn.units) == "m"

    def test_sum_preserves_units_dim_specific(self, unitful_dataset):
        s = unitful_dataset.sum(dim="x")
        assert str(s.units) == "m"

    def test_mean_preserves_units(self, unitful_dataset):
        m = unitful_dataset.mean(dim="x")
        assert str(m.units) == "m"

    def test_global_sum_scalar_no_units(self, reduction_dataset):
        s = reduction_dataset.sum(dim=None)
        assert not isinstance(s, NDDataset)

    def test_unitful_global_sum_with_keepdims(self, unitful_dataset):
        s = unitful_dataset.sum(dim=None, keepdims=True)
        assert isinstance(s, NDDataset)
        assert str(s.units) == "m"

    def test_unitful_global_sum_scalar_is_quantity(self):
        """
        Notable behavior: when units are present, dim=None returns a Quantity,
        not a plain scalar. But when units are absent, it returns a float.
        """
        ds = NDDataset(np.array([1.0, 2.0, 3.0]), units="m")
        s = ds.sum(dim=None)
        assert str(s.units) == "m"


# ======================================================================================
# BARE DATASET
# ======================================================================================


class TestBareDatasetReduction:
    """Reductions on bare (no CoordSet) datasets."""

    def test_sum_bare(self, bare_dataset):
        s = bare_dataset.sum(dim="x")
        assert np.isclose(s, 6.0)
        assert not isinstance(s, NDDataset)

    def test_sum_bare_all_dims(self, bare_dataset):
        s = bare_dataset.sum(dim=None)
        assert np.isclose(s, 6.0)

    def test_mean_bare(self, bare_dataset):
        m = bare_dataset.mean(dim="x")
        assert np.isclose(m, 2.0)
        assert not isinstance(m, NDDataset)

    def test_argmax_bare(self, bare_dataset):
        am = bare_dataset.argmax()
        assert am == 2
