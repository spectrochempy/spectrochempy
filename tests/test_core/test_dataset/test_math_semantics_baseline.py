# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Characterization tests for mathematical semantics.

This suite characterizes the mathematical semantics of NDDataset arithmetic
and ufunc application.  Since the arithmetic-title-semantics policy has been
accepted, the title assertions in STEP 8 validate the normative behavior of
the policy (RFC sections 5.1, 5.5, 6 and 7); the rest of the suite
characterizes current behavior to detect future changes.

Coverage:
    - arithmetic (dataset +/-/*/ scalar)
    - arithmetic (dataset +/- dataset)
    - ufuncs (abs, sqrt, exp, log, sin)
    - title semantics (identity-preserving, unary composition, dataset-scalar,
      dataset-dataset, canonical scalars, growth rule, absent titles, in-place
      operators, out-of-norm operations)
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
from tests.test_core.test_dataset._semantic_dataset_helpers import assert_dims_equal
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_history_appended,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import assert_masked_values
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_units_preserved,
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
        assert_units_preserved(rich_dataset + 2.0, rich_dataset)

    def test_sub_units(self, rich_dataset):
        assert_units_preserved(rich_dataset - 2.0, rich_dataset)

    def test_mul_units(self, rich_dataset):
        assert_units_preserved(rich_dataset * 2.0, rich_dataset)

    def test_truediv_units(self, rich_dataset):
        assert_units_preserved(rich_dataset / 2.0, rich_dataset)

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
        assert_dims_equal(rich_dataset + 2.0, ["y", "x"])

    def test_sub_dims(self, rich_dataset):
        assert_dims_equal(rich_dataset - 2.0, ["y", "x"])

    def test_mul_dims(self, rich_dataset):
        assert_dims_equal(rich_dataset * 2.0, ["y", "x"])

    def test_truediv_dims(self, rich_dataset):
        assert_dims_equal(rich_dataset / 2.0, ["y", "x"])

    # ---- mask ----

    def test_add_mask(self, rich_dataset):
        assert_masked_values(rich_dataset + 2.0, (0, 0))

    def test_sub_mask(self, rich_dataset):
        assert_masked_values(rich_dataset - 2.0, (0, 0))

    def test_mul_mask(self, rich_dataset):
        assert_masked_values(rich_dataset * 2.0, (0, 0))

    def test_truediv_mask(self, rich_dataset):
        assert_masked_values(rich_dataset / 2.0, (0, 0))

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
        assert_history_appended(
            rich_dataset + 2.0, rich_dataset, "Binary operation add"
        )

    def test_sub_history(self, rich_dataset):
        assert_history_appended(
            rich_dataset - 2.0, rich_dataset, "Binary operation sub"
        )

    def test_mul_history(self, rich_dataset):
        assert_history_appended(
            rich_dataset * 2.0, rich_dataset, "Binary operation mul"
        )

    def test_truediv_history(self, rich_dataset):
        assert_history_appended(
            rich_dataset / 2.0, rich_dataset, "Binary operation truediv"
        )

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
        - Title follows T5: identical titles preserved, different titles
          composed (``add(...)`` / ``subtract(...)``)
        - Name preserved from first operand
        - History appended with binary operation info
    """

    def test_add_return_type(self, rich_dataset, compatible_dataset):
        assert isinstance(rich_dataset + compatible_dataset, NDDataset)

    def test_sub_return_type(self, rich_dataset, compatible_dataset):
        assert isinstance(rich_dataset - compatible_dataset, NDDataset)

    def test_add_units(self, rich_dataset, compatible_dataset):
        assert_units_preserved(rich_dataset + compatible_dataset, rich_dataset)

    def test_sub_units(self, rich_dataset, compatible_dataset):
        assert_units_preserved(rich_dataset - compatible_dataset, rich_dataset)

    def test_add_coordset_preserved(self, rich_dataset, compatible_dataset):
        assert_coordset_matches(rich_dataset + compatible_dataset, rich_dataset)

    def test_sub_coordset_preserved(self, rich_dataset, compatible_dataset):
        assert_coordset_matches(
            rich_dataset - compatible_dataset, rich_dataset, dims=("y",)
        )

    def test_add_dims(self, rich_dataset, compatible_dataset):
        assert_dims_equal(rich_dataset + compatible_dataset, ["y", "x"])

    def test_sub_dims(self, rich_dataset, compatible_dataset):
        assert_dims_equal(rich_dataset - compatible_dataset, ["y", "x"])

    def test_add_title(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).title == (
            "add(reference spectrum, compatible spectrum)"
        )

    def test_sub_title(self, rich_dataset, compatible_dataset):
        assert (rich_dataset - compatible_dataset).title == (
            "subtract(reference spectrum, compatible spectrum)"
        )

    def test_add_name(self, rich_dataset, compatible_dataset):
        assert (rich_dataset + compatible_dataset).name == "rich_dataset"

    def test_sub_name(self, rich_dataset, compatible_dataset):
        assert (rich_dataset - compatible_dataset).name == "rich_dataset"

    def test_add_history(self, rich_dataset, compatible_dataset):
        assert_history_appended(
            rich_dataset + compatible_dataset, rich_dataset, "Binary operation add"
        )

    def test_sub_history(self, rich_dataset, compatible_dataset):
        assert_history_appended(
            rich_dataset - compatible_dataset, rich_dataset, "Binary operation sub"
        )

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
        assert_masked_values(rich_dataset + compatible_dataset, (0, 0))

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
        abs    -- identity-preserving (T1), title kept verbatim
        sqrt   -- domain changing (T2/T3), title composed
        exp    -- requires dimensionless units
        log    -- requires dimensionless units
        sin    -- domain changing (T2/T3), title composed

    Each ufunc is tested with appropriate unit context.
    """

    # ---- abs ----

    def test_abs_return_type(self, rich_dataset):
        assert isinstance(abs(rich_dataset), NDDataset)

    def test_abs_units(self, rich_dataset):
        assert_units_preserved(abs(rich_dataset), rich_dataset)

    def test_abs_title(self, rich_dataset):
        assert abs(rich_dataset).title == "reference spectrum"

    def test_abs_name(self, rich_dataset):
        assert abs(rich_dataset).name == "rich_dataset"

    def test_abs_coordset(self, rich_dataset):
        assert_coordset_matches(abs(rich_dataset), rich_dataset)

    def test_abs_dims(self, rich_dataset):
        assert_dims_equal(abs(rich_dataset), ["y", "x"])

    def test_abs_mask(self, rich_dataset):
        assert_masked_values(abs(rich_dataset), (0, 0))

    def test_abs_history(self, rich_dataset):
        assert_history_appended(abs(rich_dataset), rich_dataset, "Unary operation abs")

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
        assert_history_appended(
            rich_dataset + 2.0, rich_dataset, "Binary operation add"
        )

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
        assert_coordset_matches(abs(rich_dataset), rich_dataset)

    def test_coordset_after_mul(self, rich_dataset):
        assert_coordset_matches(rich_dataset * 2.0, rich_dataset)

    def test_coordset_after_dataset_dataset_add(self, rich_dataset, compatible_dataset):
        assert_coordset_matches(rich_dataset + compatible_dataset, rich_dataset)

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
# STEP 8: ARITHMETIC TITLE SEMANTICS
# ======================================================================================
#
# The accepted arithmetic-title-semantics policy (RFC sections 5.1, 5.5, 6
# and 7) is implemented by a single shared rule engine: operators and ufuncs
# must produce identical titles.  These tests assert the normative families:
#   T1/T1b  identity-preserving operations keep the title verbatim;
#   T2/T3   unary transforms compose ``opname(source)``;
#   T4      dataset<->scalar additive / dimensionless-scaling keep the title,
#           powers compose ``power(source, canon(p))``, reflected division and
#           reflected powers compose;
#   T5      dataset-dataset additive composes different titles, products and
#           ratios always compose;
#   growth  composed titles beyond the 120-code-point limit collapse to an
#           absent title (stored ``_title`` is ``None``).
#


class TestTitleIdentityPreserving:
    """T1/T1b: identity-preserving operations keep the source title verbatim."""

    @pytest.mark.parametrize(
        "op",
        [
            np.negative,
            np.positive,
            np.absolute,
            np.abs,
            np.fabs,
            np.rint,
            np.floor,
            np.ceil,
            np.trunc,
        ],
    )
    def test_ufunc_keeps_title(self, op):
        ds = NDDataset([1.5, -2.5], units="m", title="alpha")
        assert op(ds).title == "alpha"

    def test_unary_operators_keep_title(self):
        ds = NDDataset([1.5, -2.5], units="m", title="alpha")
        assert (+ds).title == "alpha"
        assert (-ds).title == "alpha"
        assert abs(ds).title == "alpha"


class TestTitleUnaryComposition:
    """T2/T3: domain-changing unary transforms compose ``opname(source)``."""

    @pytest.mark.parametrize(
        "op, expected",
        [
            (np.square, "square(alpha)"),
            (np.sqrt, "sqrt(alpha)"),
            (np.reciprocal, "reciprocal(alpha)"),
            (np.exp, "exp(alpha)"),
            (np.exp2, "exp2(alpha)"),
            (np.expm1, "expm1(alpha)"),
            (np.log, "log(alpha)"),
            (np.log2, "log2(alpha)"),
            (np.log10, "log10(alpha)"),
            (np.log1p, "log1p(alpha)"),
            (np.sin, "sin(alpha)"),
            (np.cos, "cos(alpha)"),
            (np.tan, "tan(alpha)"),
            (np.arcsin, "arcsin(alpha)"),
            (np.arccos, "arccos(alpha)"),
            (np.arctan, "arctan(alpha)"),
            (np.sinh, "sinh(alpha)"),
            (np.cosh, "cosh(alpha)"),
            (np.tanh, "tanh(alpha)"),
            (np.arcsinh, "arcsinh(alpha)"),
            (np.arccosh, "arccosh(alpha)"),
            (np.arctanh, "arctanh(alpha)"),
        ],
    )
    def test_ufunc_composes_title(self, op, expected):
        ds = NDDataset([0.5], units="dimensionless", title="alpha")
        assert op(ds).title == expected


class TestTitleDatasetScalar:
    """T4: dataset<->scalar additive and dimensionless-scaling keep the title."""

    def test_ufuncs_with_scalar_keep_title(self):
        ds = NDDataset([1.0, 2.0], units="absorbance", title="alpha")
        assert np.add(ds, 2.0).title == "alpha"
        assert np.subtract(ds, 2.0).title == "alpha"
        assert np.multiply(ds, 2.0).title == "alpha"
        assert np.true_divide(ds, 2.0).title == "alpha"

    def test_power_composes_canonical_scalar(self):
        ds = NDDataset([1.0, 2.0], units="absorbance", title="alpha")
        assert (ds**2).title == "power(alpha, 2)"
        assert (ds**2.0).title == "power(alpha, 2)"
        assert np.power(ds, 2).title == "power(alpha, 2)"

    def test_reflected_division_composes(self):
        ds = NDDataset([1.0, 2.0], units="absorbance", title="alpha")
        assert (2.0 / ds).title == "divide(2, alpha)"
        assert np.true_divide(2.0, ds).title == "divide(2, alpha)"

    def test_reflected_power_composes(self):
        ds = NDDataset([1.0, 2.0], units="absorbance", title="alpha")
        assert (2.0**ds).title == "power(2, alpha)"


class TestTitleDatasetDataset:
    """T5: additive composes different titles, products/ratios always compose."""

    def test_identical_titles_kept(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="alpha")
        assert (a + b).title == "alpha"
        assert (a - b).title == "alpha"

    def test_add_composes(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert (a + b).title == "add(alpha, beta)"
        assert np.add(a, b).title == "add(alpha, beta)"

    def test_subtract_composes(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert (a - b).title == "subtract(alpha, beta)"
        assert np.subtract(a, b).title == "subtract(alpha, beta)"

    def test_multiply_composes(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert (a * b).title == "multiply(alpha, beta)"
        assert np.multiply(a, b).title == "multiply(alpha, beta)"

    def test_divide_composes(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert (a / b).title == "divide(alpha, beta)"
        assert np.true_divide(a, b).title == "divide(alpha, beta)"

    def test_floor_divide_absent(self):
        a = NDDataset([4.0, 9.0], title="alpha")
        b = NDDataset([2.0, 3.0], title="beta")
        assert (a // b)._title is None
        assert np.floor_divide(a, b)._title is None

    def test_mod_absent(self):
        a = NDDataset([4.0, 9.0], title="alpha")
        b = NDDataset([2.0, 3.0], title="beta")
        assert np.mod(a, b)._title is None
        assert np.remainder(a, b)._title is None

    def test_fmod_absent(self):
        a = NDDataset([4.0, 9.0], title="alpha")
        b = NDDataset([2.0, 3.0], title="beta")
        assert np.fmod(a, b)._title is None

    def test_unlisted_binary_ufuncs_absent(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert np.maximum(a, b)._title is None
        assert np.minimum(a, b)._title is None
        assert np.fmax(a, b)._title is None
        assert np.fmin(a, b)._title is None
        assert np.copysign(a, b)._title is None
        assert np.logaddexp(a, b)._title is None
        assert np.logaddexp2(a, b)._title is None


class TestTitleOperatorUfuncParity:
    """Operators and ufuncs share one engine: identical composed titles."""

    def test_add_parity(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert (a + b).title == np.add(a, b).title

    def test_subtract_parity(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert (a - b).title == np.subtract(a, b).title

    def test_multiply_parity(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert (a * b).title == np.multiply(a, b).title == "multiply(alpha, beta)"

    def test_divide_parity(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="beta")
        assert (a / b).title == np.true_divide(a, b).title == "divide(alpha, beta)"

    def test_reflected_divide_parity(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        assert (2.0 / a).title == np.true_divide(2.0, a).title == "divide(2, alpha)"

    def test_power_parity(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds**2).title == np.power(ds, 2).title == "power(alpha, 2)"


class TestTitleCanonicalScalars:
    """Section 5.5: canonical scalar representation inside composed titles."""

    def test_integral_float_rendered_as_int(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds ** np.float64(2.0)).title == "power(alpha, 2)"
        assert (ds ** np.int64(2)).title == "power(alpha, 2)"

    def test_fractional_float_kept(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds**2.5).title == "power(alpha, 2.5)"

    def test_negative_exponent(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds**-2).title == "power(alpha, -2)"

    def test_negative_zero_rendered_as_zero(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds**-0.0).title == "power(alpha, 0)"

    def test_bool_rendered_as_int(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds ** np.bool_(True)).title == "power(alpha, 1)"

    def test_complex_with_zero_imag(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds ** (2 + 0j)).title == "power(alpha, 2)"

    def test_complex_kept(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds ** (2 + 3j)).title == "power(alpha, (2+3j))"

    def test_non_finite(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert np.power(ds, np.nan).title == "power(alpha, nan)"
        assert np.power(ds, np.inf).title == "power(alpha, inf)"

    def test_scientific_notation(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert np.true_divide(1e-05, ds).title == "divide(1e-05, alpha)"


class TestTitleGrowthAndAbsence:
    """Section 6 (120-code-point limit) and absent-title rule (section 7)."""

    def test_exact_limit_kept(self):
        source = "x" * 115
        ds = NDDataset([1.0], title=source)
        result = np.sin(ds)
        assert result.title == f"sin({source})"
        assert len(result.title) == 120

    def test_over_limit_collapses(self):
        source = "x" * 116
        ds = NDDataset([1.0], title=source)
        result = np.sin(ds)
        assert result._title is None
        assert result.title == "<untitled>"

    def test_chained_composition_collapses(self):
        ds = NDDataset([0.5], title="x")
        for _ in range(30):
            ds = np.sin(ds)
        assert ds._title is None

    def test_untitled_source_absent(self):
        ds = NDDataset([1.0, 2.0])
        assert np.sqrt(ds)._title is None
        assert np.sin(ds).title == "<untitled>"

    def test_untitled_operand_absent(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0])
        assert (a + b)._title is None
        assert (a * b)._title is None


class TestTitleOutOfNorm:
    """Operations (or operands) outside the normed table yield an absent title."""

    def test_unlisted_unary_ufunc_absent(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert np.cbrt(ds)._title is None
        assert np.conjugate(ds)._title is None

    def test_fix_preserves_title(self):
        with pytest.warns(DeprecationWarning):
            ds = NDDataset([1.0, 2.0], title="alpha")
            assert np.fix(ds).title == "alpha"

    def test_floor_divide_scalar_absent(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds // 2)._title is None
        assert np.floor_divide(ds, 2)._title is None

    def test_remainder_scalar_absent(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert np.remainder(ds, 2)._title is None

    def test_maximum_minimum_scalar_absent(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert np.maximum(ds, 2.0)._title is None
        assert np.minimum(ds, 2.0)._title is None

    def test_array_operand_keeps_title(self):
        # A non-canonizable array operand contributes no quantity identity and
        # is classified like a T4 scalar: the titled operand's context is kept
        # (explicit out-of-RFC-scope classification, PR #1534).
        ds = NDDataset([1.0, 2.0], title="alpha")
        assert (ds * np.array([1.0, 2.0])).title == "alpha"
        assert (ds + np.array([1.0, 2.0])).title == "alpha"
        assert np.multiply(ds, np.array([1.0, 2.0])).title == "alpha"
        assert np.add(ds, np.array([1.0, 2.0])).title == "alpha"


class TestTitleInplaceOperations:
    """In-place operators share the engine: ``ds **= 2`` == ``ds = ds ** 2``."""

    def test_inplace_power_composes(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        ds **= 2
        assert ds.title == "power(alpha, 2)"

    def test_inplace_additive_keeps_title(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        ds += 2
        assert ds.title == "alpha"
        ds -= 1
        assert ds.title == "alpha"

    def test_inplace_ratio_keeps_title(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        ds *= 2
        assert ds.title == "alpha"
        ds /= 2
        assert ds.title == "alpha"

    def test_inplace_floor_divide_absent(self):
        ds = NDDataset([1.0, 2.0], title="alpha")
        ds //= 2
        assert ds._title is None

    def test_inplace_matches_operator(self):
        a = NDDataset([1.0, 2.0], title="alpha")
        b = NDDataset([1.0, 2.0], title="alpha")
        b **= 2
        assert b.title == (a**2).title


class TestReflectedPowerNonMutation:
    """Reflected power (``2 ** ds``) never mutates the source dataset."""

    def test_source_not_mutated(self):
        ds = NDDataset([2.0, 4.0, 8.0], title="alpha")
        orig = ds.data.copy()
        result = 2.0**ds
        assert_array_equal(ds.data, orig)
        assert result.title == "power(2, alpha)"

    def test_result_computes_exp(self):
        ds = NDDataset([1.0, 2.0, 3.0], title="alpha")
        result = 2.0**ds
        np.testing.assert_allclose(result.data, 2.0 ** np.array([1.0, 2.0, 3.0]))


# ======================================================================================
# STEP 9: SURPRISING BEHAVIOR DOCUMENTATION
# ======================================================================================
#
# Surprising behaviors discovered during characterization:
#
# 1. Setting MASKED on a 2D NDDataset masks the ENTIRE row and column
#    intersecting at the given index.  This differs from numpy's per-element
#    masked array behavior and is the current spectroscopy-oriented policy.
#
# 2. Title follows the accepted arithmetic-title-semantics policy through a
#    single shared rule engine: operators and ufuncs produce identical titles.
#    Identity-preserving operations (T1/T1b) and dataset-scalar additive and
#    dimensionless-scaling operations (T4) keep the title verbatim; unary
#    transforms compose ``opname(source)`` (T2/T3); dataset-dataset additive
#    operations compose different titles (T5); products and ratios always
#    compose.  Operations outside the normed table (maximum, floor_divide,
#    cbrt, ...) yield an absent title (stored ``_title`` is ``None``); nothing
#    is preserved or fabricated silently.  Composed titles longer than 120 code
#    points also collapse to absent.  In-place operators share the engine
#    through their canonical names (``ds **= 2`` == ``ds = ds ** 2``); Coord
#    arithmetic is left to its own (CoordSet) title conventions.  A
#    non-canonizable array operand (e.g. ``ds + np.array([...])``) is
#    classified like a T4 scalar and keeps the titled operand's context; this
#    is an explicit out-of-RFC-scope classification, not a silent default.
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
