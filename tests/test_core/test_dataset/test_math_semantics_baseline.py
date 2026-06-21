# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Characterization tests for mathematical semantics baseline.

This suite characterizes CURRENT behavior of identity-preserving operations
on NDDataset.  It does NOT validate a desired future policy.

Purpose: detect future changes, not enforce a preferred policy.

Coverage:
    - arithmetic (dataset +/-/*/ scalar)
    - arithmetic (dataset +/- dataset)
    - ufuncs (abs, sqrt, exp, log, sin)
    - history behavior
    - CoordSet preservation
    - metadata, units, masks, title, name
"""

import numpy as np
import pytest

from spectrochempy import DimensionalityError
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Unit
from spectrochempy.utils.testing import assert_array_equal
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_basic_metadata_preserved,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_coordset_matches,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    make_semantic_2d_dataset,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import mask_dataset_points

# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def rich_dataset():
    """
    Synthetic NDDataset with all semantic features used by this suite.

    Fields and why each is included:
        data             -- must have numeric values for arithmetic
        units            -- "absorbance": physical dimension for unit checks
        title            -- identity / labelling for plots
        name             -- workflow identity
        author           -- provenance / attribution
        description      -- scientific context
        origin           -- provenance / lineage
        meta             -- custom user metadata (not a standard field)
        CoordSet         -- geometry contract for both dims (y, x)
        dims             -- named dimensions
        mask             -- some invalid values
        history          -- initial provenance marker
    """
    ds = make_semantic_2d_dataset(
        units="absorbance",
        title="reference spectrum",
        name="rich_dataset",
        description="Synthetic dataset for semantic characterization",
        meta_project="semantic_characterization",
        meta_instrument="test_instrument",
        history="Initial creation for semantic baseline",
    )

    # Mask a few values.
    # NOTE: setting MASKED on a 2D NDDataset masks the entire row AND column
    # intersecting at the given index.  This differs from numpy's per-element
    # masked array behavior and is the current spectroscopy-oriented policy.
    return mask_dataset_points(ds, (0, 0), (2, 3), (4, 6))


@pytest.fixture
def unmasked_dataset():
    """Minimal dataset without mask for numerical correctness checks."""
    return make_semantic_2d_dataset(
        title="unmasked",
        name="unmasked_dataset",
        units="absorbance",
        description="Unmasked dataset for numerical checks",
        meta_project="numeric_checks",
        history="Unmasked dataset creation",
    )


@pytest.fixture
def compatible_dataset():
    """
    Second dataset with same coordinate geometry for dataset-dataset arithmetic.

    Same shape, same coordinate values -- compatible for arithmetic.
    """
    return make_semantic_2d_dataset(
        data=np.ones((5, 7), dtype="float64") * 10.0,
        units="absorbance",
        title="compatible spectrum",
        name="compatible_dataset",
        author="compat_author",
        description="Compatible dataset for binary arithmetic",
        origin="compat_origin",
        meta_project="compat_project",
        history="Compatible dataset creation",
    )


# ======================================================================================
# STEP 1: REFERENCE DATASET PROPERTIES
# ======================================================================================


class TestReferenceDataset:
    """Verify the rich_dataset fixture has expected properties."""

    def test_rich_dataset_shape_and_type(self, rich_dataset):
        assert rich_dataset.shape == (5, 7)
        assert isinstance(rich_dataset, NDDataset)

    def test_rich_dataset_units(self, rich_dataset):
        assert rich_dataset.units == Unit("absorbance")

    def test_rich_dataset_title(self, rich_dataset):
        assert rich_dataset.title == "reference spectrum"

    def test_rich_dataset_name(self, rich_dataset):
        assert rich_dataset.name == "rich_dataset"

    def test_rich_dataset_author(self, rich_dataset):
        assert rich_dataset.author == "test_author"

    def test_rich_dataset_description(self, rich_dataset):
        assert rich_dataset.description == (
            "Synthetic dataset for semantic characterization"
        )

    def test_rich_dataset_origin(self, rich_dataset):
        assert rich_dataset.origin == "test_origin"

    def test_rich_dataset_meta(self, rich_dataset):
        assert rich_dataset.meta.project == "semantic_characterization"
        assert rich_dataset.meta.instrument == "test_instrument"

    def test_rich_dataset_dims(self, rich_dataset):
        assert rich_dataset.dims == ["y", "x"]

    def test_rich_dataset_coordset(self, rich_dataset):
        assert rich_dataset.coordset is not None
        assert len(rich_dataset.coordset) == 2

    def test_rich_dataset_mask(self, rich_dataset):
        assert rich_dataset.is_masked
        assert bool(rich_dataset.mask[0, 0])
        assert bool(rich_dataset.mask[2, 3])
        assert bool(rich_dataset.mask[4, 6])

    def test_rich_dataset_history(self, rich_dataset):
        assert len(rich_dataset.history) == 1
        assert "Initial creation for semantic baseline" in str(rich_dataset.history[0])


# ======================================================================================
# STEP 2: DATASET-SCALAR ARITHMETIC
# ======================================================================================


class TestDatasetScalarArithmetic:
    """
    Characterize dataset +/-/*/ scalar.

    Current behavior (observed):
        - Returns NDDataset
        - Preserves units
        - Preserves CoordSet
        - Preserves dims
        - Preserves mask
        - Preserves title for all four operators (+/-/*//)
        - Preserves name
        - Appends history
        - Preserves metadata (author, description, origin, meta)
    """

    # ---- return type ----

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
    def test_return_type(self, rich_dataset, op):
        scalar = 2.0
        result = getattr(rich_dataset, f"__{op}__")(scalar)
        assert isinstance(result, NDDataset)

    # ---- units ----

    def test_add_units(self, rich_dataset):
        assert (rich_dataset + 2.0).units == Unit("absorbance")

    def test_sub_units(self, rich_dataset):
        assert (rich_dataset - 2.0).units == Unit("absorbance")

    def test_mul_units(self, rich_dataset):
        assert (rich_dataset * 2.0).units == Unit("absorbance")

    def test_truediv_units(self, rich_dataset):
        assert (rich_dataset / 2.0).units == Unit("absorbance")

    # ---- CoordSet ----

    def test_add_coordset(self, rich_dataset):
        assert_coordset_matches(rich_dataset + 2.0, rich_dataset)

    def test_sub_coordset(self, rich_dataset):
        assert_coordset_matches(rich_dataset - 2.0, rich_dataset)

    def test_mul_coordset(self, rich_dataset):
        assert_coordset_matches(rich_dataset * 2.0, rich_dataset)

    def test_truediv_coordset(self, rich_dataset):
        assert_coordset_matches(rich_dataset / 2.0, rich_dataset)

    # ---- dims ----

    def test_add_dims(self, rich_dataset):
        assert (rich_dataset + 2.0).dims == ["y", "x"]

    def test_sub_dims(self, rich_dataset):
        assert (rich_dataset - 2.0).dims == ["y", "x"]

    def test_mul_dims(self, rich_dataset):
        assert (rich_dataset * 2.0).dims == ["y", "x"]

    def test_truediv_dims(self, rich_dataset):
        assert (rich_dataset / 2.0).dims == ["y", "x"]

    # ---- mask ----

    def test_add_mask(self, rich_dataset):
        result = rich_dataset + 2.0
        assert result.is_masked
        assert bool(result.mask[0, 0])

    def test_sub_mask(self, rich_dataset):
        assert (rich_dataset - 2.0).is_masked

    def test_mul_mask(self, rich_dataset):
        assert (rich_dataset * 2.0).is_masked

    def test_truediv_mask(self, rich_dataset):
        assert (rich_dataset / 2.0).is_masked

    # ---- title behavior ----
    # Current: ALL Python operators (+, -, *, /) preserve the title.
    # Title overwriting only happens via __array_ufunc__ for ufunc paths.

    def test_add_title(self, rich_dataset):
        assert (rich_dataset + 2.0).title == "reference spectrum"

    def test_sub_title(self, rich_dataset):
        assert (rich_dataset - 2.0).title == "reference spectrum"

    def test_mul_title(self, rich_dataset):
        assert (rich_dataset * 2.0).title == "reference spectrum"

    def test_truediv_title(self, rich_dataset):
        assert (rich_dataset / 2.0).title == "reference spectrum"

    # ---- name behavior ----

    def test_add_name(self, rich_dataset):
        assert (rich_dataset + 2.0).name == "rich_dataset"

    def test_sub_name(self, rich_dataset):
        assert (rich_dataset - 2.0).name == "rich_dataset"

    def test_mul_name(self, rich_dataset):
        assert (rich_dataset * 2.0).name == "rich_dataset"

    def test_truediv_name(self, rich_dataset):
        assert (rich_dataset / 2.0).name == "rich_dataset"

    # ---- history behavior ----

    def test_add_history(self, rich_dataset):
        result = rich_dataset + 2.0
        assert len(result.history) == 2
        assert "Initial creation" in str(result.history[0])
        assert "Binary operation add" in str(result.history[1])

    def test_sub_history(self, rich_dataset):
        result = rich_dataset - 2.0
        assert len(result.history) == 2
        assert "Binary operation sub" in str(result.history[1])

    def test_mul_history(self, rich_dataset):
        result = rich_dataset * 2.0
        assert len(result.history) == 2
        assert "Binary operation mul" in str(result.history[1])

    def test_truediv_history(self, rich_dataset):
        result = rich_dataset / 2.0
        assert len(result.history) == 2
        assert "Binary operation truediv" in str(result.history[1])

    # ---- metadata preservation ----

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
    def test_author_preserved(self, rich_dataset, op):
        assert_basic_metadata_preserved(
            getattr(rich_dataset, f"__{op}__")(2.0),
            rich_dataset,
            check_filename=False,
            meta_keys=("project", "instrument"),
        )

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
    def test_description_preserved(self, rich_dataset, op):
        assert_basic_metadata_preserved(
            getattr(rich_dataset, f"__{op}__")(2.0),
            rich_dataset,
            check_filename=False,
            meta_keys=("project", "instrument"),
        )

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
    def test_origin_preserved(self, rich_dataset, op):
        assert_basic_metadata_preserved(
            getattr(rich_dataset, f"__{op}__")(2.0),
            rich_dataset,
            check_filename=False,
            meta_keys=("project", "instrument"),
        )

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
    def test_meta_preserved(self, rich_dataset, op):
        assert_basic_metadata_preserved(
            getattr(rich_dataset, f"__{op}__")(2.0),
            rich_dataset,
            check_filename=False,
            meta_keys=("project", "instrument"),
        )

    # ---- numerical correctness (unmasked dataset) ----

    def test_add_numerical(self, unmasked_dataset):
        result = unmasked_dataset + 2.0
        assert_array_equal(result.data, unmasked_dataset.data + 2.0)

    def test_sub_numerical(self, unmasked_dataset):
        result = unmasked_dataset - 2.0
        assert_array_equal(result.data, unmasked_dataset.data - 2.0)

    def test_mul_numerical(self, unmasked_dataset):
        result = unmasked_dataset * 2.0
        assert_array_equal(result.data, unmasked_dataset.data * 2.0)

    def test_truediv_numerical(self, unmasked_dataset):
        result = unmasked_dataset / 2.0
        assert_array_equal(result.data, unmasked_dataset.data / 2.0)


# ======================================================================================
# STEP 3: DATASET-DATASET ARITHMETIC
# ======================================================================================


class TestDatasetDatasetArithmetic:
    """
    Characterize dataset + dataset and dataset - dataset.

    Current behavior (observed):
        - Returns NDDataset
        - Result units: first operand's units
        - CoordSet preserved from first operand
        - Metadata propagated from first operand
        - Title preserved for add/sub
        - Name preserved from first operand
        - History appended with binary operation info
    """

    def test_add_return_type(self, rich_dataset, compatible_dataset):
        assert isinstance(rich_dataset + compatible_dataset, NDDataset)

    def test_sub_return_type(self, rich_dataset, compatible_dataset):
        assert isinstance(rich_dataset - compatible_dataset, NDDataset)

    def test_add_units(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).units == Unit("absorbance")

    def test_sub_units(self, rich_dataset, compatible_dataset):
        assert (rich_dataset - compatible_dataset).units == Unit("absorbance")

    def test_add_coordset_preserved(self, rich_dataset, compatible_dataset):
        assert_coordset_matches(rich_dataset + compatible_dataset, rich_dataset)

    def test_sub_coordset_preserved(self, rich_dataset, compatible_dataset):
        assert_coordset_matches(
            rich_dataset - compatible_dataset, rich_dataset, dims=("y",)
        )

    def test_add_dims(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).dims == ["y", "x"]

    def test_sub_dims(self, rich_dataset, compatible_dataset):
        assert (rich_dataset - compatible_dataset).dims == ["y", "x"]

    def test_add_title(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).title == "reference spectrum"

    def test_sub_title(self, rich_dataset, compatible_dataset):
        assert (rich_dataset - compatible_dataset).title == "reference spectrum"

    def test_add_name(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).name == "rich_dataset"

    def test_sub_name(self, rich_dataset, compatible_dataset):
        assert (rich_dataset - compatible_dataset).name == "rich_dataset"

    def test_add_history(self, rich_dataset, compatible_dataset):
        result = rich_dataset + compatible_dataset
        assert len(result.history) == 2
        assert "Initial creation" in str(result.history[0])

    def test_sub_history(self, rich_dataset, compatible_dataset):
        result = rich_dataset - compatible_dataset
        assert len(result.history) == 2

    # ---- metadata from first operand ----

    def test_add_author(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).author == "test_author"

    def test_sub_author(self, rich_dataset, compatible_dataset):
        assert (rich_dataset - compatible_dataset).author == "test_author"

    def test_add_origin(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).origin == "test_origin"

    def test_add_meta(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).meta.project == (
            "semantic_characterization"
        )

    # ---- numerical (unmasked) ----

    def test_add_numerical(self, unmasked_dataset, compatible_dataset):
        result = unmasked_dataset + compatible_dataset
        assert_array_equal(result.data, unmasked_dataset.data + compatible_dataset.data)

    def test_sub_numerical(self, unmasked_dataset, compatible_dataset):
        result = unmasked_dataset - compatible_dataset
        assert_array_equal(result.data, unmasked_dataset.data - compatible_dataset.data)

    # ---- mask: union of both masks ----

    def test_add_mask_union(self, rich_dataset, compatible_dataset):
        result = rich_dataset + compatible_dataset
        assert result.is_masked
        assert bool(result.mask[0, 0])

    # ---- reverse order (scalar on right, unmasked) ----

    def test_radd_numerical(self, unmasked_dataset):
        assert_array_equal((2.0 + unmasked_dataset).data, 2.0 + unmasked_dataset.data)

    def test_rsub_numerical(self, unmasked_dataset):
        assert_array_equal(
            (100.0 - unmasked_dataset).data, 100.0 - unmasked_dataset.data
        )

    def test_rmul_numerical(self, unmasked_dataset):
        assert_array_equal((2.0 * unmasked_dataset).data, 2.0 * unmasked_dataset.data)


# ======================================================================================
# STEP 4: UFUNC CHARACTERIZATION
# ======================================================================================


class TestUfuncCharacterization:
    """
    Characterize representative ufuncs on NDDataset.

    Selected operations:
        abs    -- identity-preserving, in __keep_title
        sqrt   -- domain changing, not in __keep_title
        exp    -- requires dimensionless units
        log    -- requires dimensionless units
        sin    -- requires radian units, returns dimensionless

    Each ufunc is tested with appropriate unit context.
    """

    # ---- abs ----

    def test_abs_return_type(self, rich_dataset):
        assert isinstance(abs(rich_dataset), NDDataset)

    def test_abs_units(self, rich_dataset):
        assert abs(rich_dataset).units == Unit("absorbance")

    def test_abs_title(self, rich_dataset):
        assert abs(rich_dataset).title == "reference spectrum"

    def test_abs_name(self, rich_dataset):
        assert abs(rich_dataset).name == "rich_dataset"

    def test_abs_coordset(self, rich_dataset):
        assert abs(rich_dataset).coordset is not None

    def test_abs_dims(self, rich_dataset):
        assert abs(rich_dataset).dims == ["y", "x"]

    def test_abs_mask(self, rich_dataset):
        result = abs(rich_dataset)
        assert result.is_masked
        assert bool(result.mask[0, 0])

    def test_abs_history(self, rich_dataset):
        result = abs(rich_dataset)
        assert len(result.history) == 2
        # Note: history message uses "abs", not "absolute"
        assert "Unary operation abs" in str(result.history[1])

    def test_abs_meta(self, rich_dataset):
        assert abs(rich_dataset).meta.project == "semantic_characterization"

    # ---- sqrt ----

    def test_sqrt_positive_data(self):
        ds = NDDataset(np.array([1.0, 4.0, 9.0, 16.0]), units="absorbance")
        result = np.sqrt(ds)
        assert isinstance(result, NDDataset)
        assert_array_equal(result.data, np.array([1.0, 2.0, 3.0, 4.0]))
        assert result.units == Unit("absorbance")

    def test_sqrt_title(self):
        ds = NDDataset([1.0, 4.0], title="test")
        result = np.sqrt(ds)
        assert result.title == "sqrt(test)"

    def test_sqrt_name(self):
        ds = NDDataset([1.0, 4.0], name="myname")
        result = np.sqrt(ds)
        assert result.name == "myname"

    # ---- exp (requires dimensionless) ----

    def test_exp_dimensionless(self):
        ds = NDDataset(np.array([0.0, 1.0, 2.0]), units="dimensionless")
        result = np.exp(ds)
        assert isinstance(result, NDDataset)
        assert result.units == Unit("dimensionless")

    def test_exp_rejects_dimensioned(self):
        ds = NDDataset([1.0, 2.0], units="m")
        with pytest.raises(DimensionalityError):
            np.exp(ds)

    # ---- log (requires dimensionless) ----

    def test_log_dimensionless(self):
        ds = NDDataset(np.array([1.0, 2.0, 3.0]), units="dimensionless")
        result = np.log(ds)
        assert isinstance(result, NDDataset)
        assert result.units == Unit("dimensionless") or result.dimensionless

    def test_log_rejects_dimensioned(self):
        ds = NDDataset(np.array([1.0, 2.0, 3.0]), units="m")
        with pytest.raises(DimensionalityError):
            np.log(ds)

    # ---- sin (requires radian) ----

    def test_sin_radian(self):
        ds = NDDataset(np.array([0.0, np.pi / 2.0]), units="radian")
        result = np.sin(ds)
        assert isinstance(result, NDDataset)
        assert result.units == Unit("dimensionless")

    def test_sin_rejects_non_angular(self):
        ds = NDDataset([1.0, 2.0], units="m")
        with pytest.raises(DimensionalityError):
            np.sin(ds)

    def test_sin_title(self):
        ds = NDDataset([0.0, 1.0], title="angle", units="radian")
        result = np.sin(ds)
        assert result.title == "sin(angle)"

    def test_sin_name(self):
        ds = NDDataset([0.0, 1.0], name="angle_data", units="radian")
        result = np.sin(ds)
        assert result.name == "angle_data"


# ======================================================================================
# STEP 5: HISTORY BEHAVIOR
# ======================================================================================


class TestHistoryBehavior:
    """
    Characterize current history behavior for various operation types.

    Questions documented:
        - Is history copied?           YES (from source dataset)
        - Is history appended?         YES (for copy-first operations)
        - Is history regenerated?      Only for wrapper/reconstruction paths
        - Does behavior differ by operation?  YES (see individual tests)
    """

    def test_history_is_list_of_strings(self, rich_dataset):
        assert isinstance(rich_dataset.history, list)
        for entry in rich_dataset.history:
            assert isinstance(entry, str)

    def test_history_timestamped(self, rich_dataset):
        for entry in rich_dataset.history:
            assert ">" in entry

    def test_history_copied_in_arithmetic(self, rich_dataset):
        result = rich_dataset + 2.0
        assert len(result.history) == 2
        assert str(result.history[0]) == str(rich_dataset.history[0])

    def test_history_appended_not_replaced(self, rich_dataset):
        result = rich_dataset * 2.0
        assert len(result.history) == 2
        assert "Binary operation" in str(result.history[1])

    def test_history_multiple_operations(self, rich_dataset):
        result = rich_dataset + 2.0
        result = result * 3.0
        result = result - 1.0
        assert len(result.history) == 4
        assert "Initial creation" in str(result.history[0])
        assert "Binary operation add" in str(result.history[1])
        assert "Binary operation mul" in str(result.history[2])
        assert "Binary operation sub" in str(result.history[3])

    def test_abs_history_message(self, rich_dataset):
        result = abs(rich_dataset)
        # Note: history message uses short form "abs", not "absolute"
        assert "Unary operation abs" in str(result.history[1])

    def test_neg_history_message(self):
        ds = NDDataset([1.0, 2.0], history="init")
        result = -ds
        assert "Unary operation neg" in str(result.history[1])

    def test_ufunc_history_message(self):
        ds = NDDataset([1.0, 4.0], history="init")
        result = np.sqrt(ds)
        # Ufunc path uses "Ufunc sqrt applied." format
        assert "Ufunc sqrt applied" in str(result.history[1])


# ======================================================================================
# STEP 6: COORDSET PRESERVATION
# ======================================================================================


class TestCoordSetPreservation:
    """
    Characterize CoordSet preservation for identity-preserving operations.

    Current observation:
        - CoordSet object is preserved (same type)
        - CoordSet.names lists dims in alphabetical order (["x", "y"])
        - Coordinate values are unchanged
        - Coordinate metadata (title, units) is preserved
    """

    def test_coordset_not_none(self, rich_dataset):
        assert rich_dataset.coordset is not None

    def test_coordset_names(self, rich_dataset):
        # NOTE: CoordSet.names returns dims in alphabetical order,
        # not in the dims order (which is ["y", "x"])
        names = rich_dataset.coordset.names
        assert sorted(names) == ["x", "y"]
        assert len(names) == 2

    def test_coordset_coord_types(self, rich_dataset):
        assert isinstance(rich_dataset.y, Coord)
        assert isinstance(rich_dataset.x, Coord)

    def test_coord_values_unchanged_add(self, rich_dataset):
        result = rich_dataset + 2.0
        np.testing.assert_allclose(result.y.data, rich_dataset.y.data)
        np.testing.assert_allclose(result.x.data, rich_dataset.x.data)

    def test_coord_units_after_add(self, rich_dataset):
        result = rich_dataset + 2.0
        assert result.y.units == rich_dataset.y.units
        assert result.x.units == rich_dataset.x.units

    def test_coord_titles_after_add(self, rich_dataset):
        result = rich_dataset + 2.0
        assert result.y.title == rich_dataset.y.title
        assert result.x.title == rich_dataset.x.title

    def test_coord_values_unchanged_abs(self, rich_dataset):
        result = abs(rich_dataset)
        np.testing.assert_allclose(result.y.data, rich_dataset.y.data)
        np.testing.assert_allclose(result.x.data, rich_dataset.x.data)

    def test_coordset_after_mul(self, rich_dataset):
        result = rich_dataset * 2.0
        assert result.coordset is not None
        np.testing.assert_allclose(result.y.data, rich_dataset.y.data)
        np.testing.assert_allclose(result.x.data, rich_dataset.x.data)

    def test_coordset_after_dataset_dataset_add(self, rich_dataset, compatible_dataset):
        result = rich_dataset + compatible_dataset
        assert result.coordset is not None
        np.testing.assert_allclose(result.y.data, rich_dataset.y.data)
        np.testing.assert_allclose(result.x.data, rich_dataset.x.data)

    def test_double_coordset_preserved(self):
        y = Coord(np.arange(4.0), title="rows")
        x1 = Coord(np.linspace(100, 500, 5), title="wavenumber")
        x2 = Coord(np.arange(5.0), title="second")
        ds = NDDataset(
            np.arange(20.0).reshape(4, 5),
            coordset=[y, CoordSet(x1, x2)],
            title="multicoord",
        )
        result = ds + 2.0
        assert result.coordset is not None
        surviving = result.coordset["x"]
        assert isinstance(surviving, CoordSet)
        assert len(surviving) == 2


# ======================================================================================
# STEP 7: ADDITIONAL SEMANTIC CHARACTERIZATION
# ======================================================================================


class TestUnaryOperators:
    """Characterize unary operator behavior (+, -, abs via operator)."""

    def test_unary_pos(self):
        ds = NDDataset(np.array([-1.0, 2.0, -3.0]), units="m", title="test")
        result = +ds
        assert isinstance(result, NDDataset)
        assert result.units == Unit("m")
        assert result.title == "test"

    def test_unary_neg(self):
        ds = NDDataset(np.array([-1.0, 2.0, -3.0]), units="m", title="test")
        result = -ds
        assert isinstance(result, NDDataset)
        assert result.units == Unit("m")
        assert result.title == "test"

    def test_unary_neg_history(self):
        ds = NDDataset([1.0, 2.0], history="created")
        result = -ds
        assert len(result.history) == 2
        assert "Unary operation neg" in str(result.history[1])


class TestInplaceOperations:
    """Characterize in-place operation behavior."""

    def test_iadd_modifies_inplace(self, unmasked_dataset):
        ds = unmasked_dataset.copy()
        ds += 2.0
        assert_array_equal(ds.data, unmasked_dataset.data + 2.0)

    def test_iadd_preserves_metadata(self, rich_dataset):
        ds = rich_dataset.copy()
        ds += 2.0
        assert ds.title == "reference spectrum"
        assert ds.name == "rich_dataset"
        assert ds.author == "test_author"

    def test_iadd_history(self, rich_dataset):
        ds = rich_dataset.copy()
        ds += 2.0
        assert len(ds.history) == 2
        assert "Inplace binary op" in str(ds.history[1])


class TestInplaceOperationsUnmasked:
    """Characterize in-place arithmetic on unmasked dataset."""

    def test_iadd_data(self, unmasked_dataset):
        ds = unmasked_dataset.copy()
        ds += 2.0
        assert_array_equal(ds.data, unmasked_dataset.data + 2.0)

    def test_isub_data(self, unmasked_dataset):
        ds = unmasked_dataset.copy()
        ds -= 2.0
        assert_array_equal(ds.data, unmasked_dataset.data - 2.0)

    def test_imul_data(self, unmasked_dataset):
        ds = unmasked_dataset.copy()
        ds *= 2.0
        assert_array_equal(ds.data, unmasked_dataset.data * 2.0)

    def test_itruediv_data(self, unmasked_dataset):
        ds = unmasked_dataset.copy()
        ds /= 2.0
        assert_array_equal(ds.data, unmasked_dataset.data / 2.0)


class TestNumericalProperties:
    """Characterize basic numerical properties after operations."""

    def test_add_preserves_shape(self, rich_dataset):
        assert (rich_dataset + 2.0).shape == rich_dataset.shape

    def test_mul_preserves_shape(self, rich_dataset):
        assert (rich_dataset * 2.0).shape == rich_dataset.shape

    def test_truediv_preserves_shape(self, rich_dataset):
        assert (rich_dataset / 2.0).shape == rich_dataset.shape

    def test_sub_preserves_shape(self, rich_dataset):
        assert (rich_dataset - 2.0).shape == rich_dataset.shape

    def test_dtype_preserved_arithmetic(self, unmasked_dataset):
        assert (unmasked_dataset + 2.0).dtype == unmasked_dataset.dtype


class TestEmptyDatasetBehavior:
    """Characterize behavior on minimal datasets."""

    def test_plain_numpy_array_ops(self):
        ds = NDDataset([1.0, 2.0, 3.0])
        result = ds + 1.0
        assert isinstance(result, NDDataset)
        assert_array_equal(result.data, [2.0, 3.0, 4.0])

    def test_no_coordset_operations(self):
        ds = NDDataset([1.0, 2.0, 3.0], units="m")
        result = ds * 2.0
        assert result.units == Unit("m")
        assert result.coordset is None

    def test_history_single_entry_on_bare_dataset(self):
        ds = NDDataset([1.0, 2.0])
        result = ds + 1.0
        assert len(result.history) == 1
        assert "Binary operation add" in str(result.history[0])


# ======================================================================================
# STEP 8: SURPRISING BEHAVIOR DOCUMENTATION
# ======================================================================================
#
# Surprising behaviors discovered during characterization:
#
# 1. Setting MASKED on a 2D NDDataset masks the ENTIRE row and column
#    intersecting at the given index.  This differs from numpy's per-element
#    masked array behavior and is the current spectroscopy-oriented policy.
#
# 2. Title is preserved for ALL Python operators (+, -, *, /) on NDDataset.
#    The __remove_title list (multiply, divide, etc.) only affects operations
#    routed through __array_ufunc__ (e.g. np.multiply()), not through Python
#    operators (__mul__, __truediv__).  This is a split between operator and
#    ufunc title behavior.
#
# 3. History messages differ by operation path:
#    - Binary operators: "Binary operation {name} with `{other}` ..."
#    - Unary operators: "Unary operation {short_name} applied"
#      where short_name is "abs", "neg", "pos", etc. (not "absolute", "negative")
#    - Ufunc via __array_ufunc__: "Ufunc {name} applied."
#    - In-place: "Inplace binary op: {name} with `{other}`"
#
# 4. CoordSet.names returns names in alphabetical order (["x", "y"]),
#    while dims are ordered as assigned (["y", "x"]).
#
# 5. For bare datasets (no history), binary operations still record history
#    because _binary_op checks `hasattr(self, "history")`.
#    Bare NDDataset has _history = [], so len(history) == 1 after one op.
#
# 6. Metadata propagation: all of author, description, origin, meta are
#    preserved by copy-first assembly for all four arithmetic operations and
#    for abs.  This matches the "copy-first" pattern described in the RFC.
#
# 7. CoordSet is always preserved for copy-first operations: the same Coord
#    objects (same values, titles, units) survive the copy.
#
# 8. Unit "absorbance" is dimensionless in pint, so np.exp() accepts it.
#    Only genuinely dimensioned units (e.g., "m") are rejected.
