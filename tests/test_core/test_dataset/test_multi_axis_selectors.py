"""
Tests for tuple/list multi-dimension selector support on the M1 reduction family.

Per the accepted RFC (dimension-selection-policy-rfc.md), multi-axis
reductions MUST accept tuple/list selectors including mixed names and
positional indices.  Single-axis reductions (argmax, argmin, cumsum,
coordmax, coordmin) MUST reject tuple/list selectors.

Validated:
    - M1 methods: mean, sum, std, var, amax, amin, ptp, all, any, average
    - M2 rejection: argmax, argmin, cumsum, coordmax, coordmin
    - Selector validation: empty, bool, duplicate, nested, conflict
    - Selector forms: tuple, list, names, ints, np.int64, negatives, mixed
    - CoordSet handling with keepdims and without
    - 3D multi-axis reductions
    - axis= keyword equivalence with dim= for tuples
    - Source non-mutation
    - Metadata preservation
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def ds2d():
    """3x4 2D dataset with dims ['y', 'x']."""
    return NDDataset(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=["y", "x"],
    )


@pytest.fixture
def ds3d():
    """2x3x4 3D dataset with dims ['z', 'y', 'x']."""
    return NDDataset(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        dims=["z", "y", "x"],
    )


@pytest.fixture
def ds2d_with_coordset():
    """2D dataset with coordinate objects attached."""
    ds = NDDataset(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=["y", "x"],
    )
    ds.set_coordset(
        y=Coord([1.0, 2.0, 3.0], title="time", units="s"),
        x=Coord([10.0, 20.0, 30.0, 40.0], title="wavenumber", units="cm^-1"),
    )
    return ds


@pytest.fixture
def ds2d_metadata():
    """2D dataset with metadata for preservation checks."""
    ds = NDDataset(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=["y", "x"],
        title="test_title",
        name="test_name",
    )
    ds.author = "test_author"
    return ds


# ======================================================================================
# M1 — MULTI-AXIS REDUCTIONS: TUPLE SELECTORS MATCH NUMPY
# ======================================================================================


class TestMultiAxisNumpyEquivalence:
    """Multi-axis reductions match numpy's result for the same axes."""

    @pytest.mark.parametrize(
        "method",
        ["mean", "sum", "std", "var", "amax", "amin", "ptp"],
    )
    def test_tuple_matches_numpy(self, ds2d, method):
        data = ds2d.data
        np_result = getattr(np, method)(data, axis=(0, 1))
        scp_result = getattr(ds2d, method)(dim=("y", "x"))
        np.testing.assert_allclose(scp_result, np_result)

    @pytest.mark.parametrize(
        "method",
        ["mean", "sum", "std", "var", "amax", "amin", "ptp"],
    )
    def test_list_matches_numpy(self, ds2d, method):
        data = ds2d.data
        np_result = getattr(np, method)(data, axis=(0, 1))
        scp_result = getattr(ds2d, method)(dim=["y", "x"])
        np.testing.assert_allclose(scp_result, np_result)

    def test_all_multi_axis(self, ds2d):
        np_result = np.all(ds2d.data, axis=(0, 1))
        scp_result = ds2d.all(dim=("y", "x"))
        np.testing.assert_array_equal(scp_result, np_result)

    def test_any_multi_axis(self, ds2d):
        np_result = np.any(ds2d.data, axis=(0, 1))
        scp_result = ds2d.any(dim=("y", "x"))
        np.testing.assert_array_equal(scp_result, np_result)

    def test_average_multi_axis(self, ds2d):
        np_result = np.average(ds2d.data, axis=(0, 1))
        scp_result = ds2d.average(dim=("y", "x"))
        np.testing.assert_allclose(scp_result, np_result)


# ======================================================================================
# M1 — SELECTOR FORMS
# ======================================================================================


class TestSelectorForms:
    """Various selector forms produce equivalent results."""

    def test_axis_keyword_tuple(self, ds2d):
        """axis= tuple is equivalent to dim= tuple."""
        r_dim = ds2d.mean(dim=("y", "x"))
        r_axis = ds2d.mean(axis=("y", "x"))
        np.testing.assert_allclose(r_dim, r_axis)

    def test_int_tuple(self, ds2d):
        """Tuple of integer indices works."""
        r = ds2d.sum(dim=(0, 1))
        assert np.isclose(r, 66.0)

    def test_negative_int_tuple(self, ds2d):
        """Tuple of negative integer indices works."""
        r = ds2d.sum(dim=(-1, -2))
        assert np.isclose(r, 66.0)

    def test_mixed_name_int(self, ds2d):
        """Mixed dimension names and integer indices in a tuple."""
        r_name = ds2d.sum(dim=("y", "x"))
        r_mixed = ds2d.sum(dim=("y", 1))
        np.testing.assert_allclose(r_name, r_mixed)

    def test_numpy_int64(self, ds2d):
        """numpy.int64 values are accepted as axis indices."""
        r = ds2d.sum(dim=(np.int64(0), np.int64(1)))
        assert np.isclose(r, 66.0)

    def test_list_form(self, ds2d):
        """List selector produces the same result as tuple selector."""
        r_tuple = ds2d.mean(dim=("y", "x"))
        r_list = ds2d.mean(dim=["y", "x"])
        np.testing.assert_allclose(
            np.asarray(r_tuple),
            np.asarray(r_list),
            rtol=1e-14,
        )


# ======================================================================================
# M1 — KEEPDIMS
# ======================================================================================


class TestMultiAxisKeepdims:
    """keepdims=True with tuple selector preserves all dimensions."""

    @pytest.mark.parametrize(
        "method",
        ["mean", "sum", "std", "var", "amax", "amin"],
    )
    def test_keepdims_preserves_all_dims(self, ds2d, method):
        r = getattr(ds2d, method)(dim=("y", "x"), keepdims=True)
        assert r.shape == (1, 1)
        assert list(r.dims) == ["y", "x"]


# ======================================================================================
# M1 — 3D REDUCTIONS
# ======================================================================================


class TestMultiAxis3D:
    """Multi-axis reductions on 3D datasets."""

    def test_reduce_two_of_three_dims(self, ds3d):
        """Reducing two of three dimensions leaves the third."""
        r = ds3d.mean(dim=("y", "x"))
        assert r.shape == (2,)
        assert list(r.dims) == ["z"]

    def test_reduce_with_int_tuple(self, ds3d):
        """Integer tuple selects axes correctly in 3D."""
        r = ds3d.mean(dim=(0, 2))
        assert r.shape == (3,)
        assert list(r.dims) == ["y"]

    def test_reduce_first_two_dims(self, ds3d):
        """Reducing first two dims."""
        r = ds3d.sum(dim=("z", "y"))
        assert r.shape == (4,)
        assert list(r.dims) == ["x"]

    def test_3d_keepdims(self, ds3d):
        """keepdims=True on 3D with two-dim reduction."""
        r = ds3d.mean(dim=("y", "x"), keepdims=True)
        assert r.shape == (2, 1, 1)
        assert list(r.dims) == ["z", "y", "x"]


# ======================================================================================
# M1 — COORDSET HANDLING
# ======================================================================================


class TestMultiAxisCoordSet:
    """CoordSet is correctly reduced for multi-axis selectors."""

    def test_full_reduction_drops_coordset(self, ds2d_with_coordset):
        """Full reduction returns a scalar (no coordset)."""
        r = ds2d_with_coordset.mean(dim=("y", "x"))
        assert not isinstance(r, NDDataset)

    def test_keepdims_preserves_coordset(self, ds2d_with_coordset):
        """keepdims=True with tuple keeps both dims and their coords."""
        r = ds2d_with_coordset.mean(dim=("y", "x"), keepdims=True)
        assert r.shape == (1, 1)
        assert list(r.dims) == ["y", "x"]
        assert r.coordset is not None

    def test_amax_coordset_tuple(self, ds2d_with_coordset):
        """Amax with tuple selector reduces coordset correctly."""
        r = ds2d_with_coordset.amax(dim=("y", "x"), keepdims=True)
        assert r.shape == (1, 1)
        assert r.coordset is not None

    def test_amin_coordset_tuple(self, ds2d_with_coordset):
        """Amin with tuple selector reduces coordset correctly."""
        r = ds2d_with_coordset.amin(dim=("y", "x"), keepdims=True)
        assert r.shape == (1, 1)

    def test_sum_coordset_tuple(self, ds2d_with_coordset):
        """Sum with tuple reduces coordset correctly."""
        r = ds2d_with_coordset.sum(dim=("y", "x"), keepdims=True)
        assert r.shape == (1, 1)
        assert list(r.dims) == ["y", "x"]

    def test_coordset_surviving_coord_title(self, ds2d_with_coordset):
        """Partial reduction keeps the surviving coord's metadata."""
        r = ds2d_with_coordset.sum(dim="y")
        assert r.coordset is not None
        assert r.coordset.names == ["x"]
        assert r["x"].title == "wavenumber"


# ======================================================================================
# SELECTOR VALIDATION
# ======================================================================================


class TestSelectorValidation:
    """Invalid selectors are rejected with clear errors."""

    def test_empty_tuple_rejected(self, ds2d):
        with pytest.raises(TypeError, match="empty sequence"):
            ds2d.mean(dim=())

    def test_empty_list_rejected(self, ds2d):
        with pytest.raises(TypeError, match="empty sequence"):
            ds2d.mean(dim=[])

    def test_bool_in_tuple_rejected(self, ds2d):
        with pytest.raises(TypeError, match="Boolean"):
            ds2d.mean(dim=(True, "x"))

    def test_bool_alone_rejected(self, ds2d):
        with pytest.raises(TypeError, match="Boolean"):
            ds2d.mean(dim=True)

    def test_duplicate_dim_rejected(self, ds2d):
        with pytest.raises(TypeError, match="Duplicate"):
            ds2d.mean(dim=("x", "x"))

    def test_duplicate_mixed_rejected(self, ds2d):
        """Same dim expressed as name and index is still a duplicate."""
        with pytest.raises(TypeError, match="Duplicate"):
            ds2d.mean(dim=("x", 1))

    def test_nested_tuple_rejected(self, ds2d):
        with pytest.raises(TypeError, match="Nested"):
            ds2d.mean(dim=(("y", "x"),))

    def test_nested_list_rejected(self, ds2d):
        with pytest.raises(TypeError, match="Nested"):
            ds2d.mean(dim=[["y", "x"]])

    def test_conflicting_dim_axis_with_tuple(self, ds2d):
        with pytest.raises(TypeError, match="conflicting"):
            ds2d.mean(dim=("y", "x"), axis="y")


# ======================================================================================
# M2 — SINGLE-AXIS REJECTION
# ======================================================================================


class TestSingleAxisRejection:
    """Single-axis methods must not accept tuple/list selectors."""

    @pytest.mark.parametrize(
        "method",
        ["argmax", "argmin", "cumsum", "coordmax", "coordmin"],
    )
    def test_rejects_tuple(self, ds2d, method):
        with pytest.raises(TypeError, match="single dimension"):
            getattr(ds2d, method)(dim=("y", "x"))

    @pytest.mark.parametrize(
        "method",
        ["argmax", "argmin", "cumsum", "coordmax", "coordmin"],
    )
    def test_rejects_list(self, ds2d, method):
        with pytest.raises(TypeError, match="single dimension"):
            getattr(ds2d, method)(dim=["y", "x"])


# ======================================================================================
# SOURCE NON-MUTATION AND METADATA
# ======================================================================================


class TestMultiAxisInvariants:
    """Multi-axis reductions preserve invariants."""

    def test_source_not_mutated(self, ds2d):
        """Multi-axis reduction does not mutate the source dataset."""
        data_before = ds2d.data.copy()
        ds2d.mean(dim=("y", "x"))
        np.testing.assert_array_equal(ds2d.data, data_before)

    def test_metadata_preserved_on_intermediate(self, ds2d_metadata):
        """Partial reduction preserves metadata."""
        r = ds2d_metadata.mean(dim="y")
        assert r.title == "test_title"
        assert r.name == "test_name"
        assert r.author == "test_author"

    def test_all_single_dim_still_works(self, ds2d):
        """Existing single-dim reductions are not broken."""
        r = ds2d.mean(dim="x")
        assert r.shape == (3,)
        assert list(r.dims) == ["y"]

    def test_global_still_works(self, ds2d):
        """Global reduction (dim=None) is unchanged."""
        r = ds2d.mean(dim=None)
        assert np.isclose(r, 5.5)
