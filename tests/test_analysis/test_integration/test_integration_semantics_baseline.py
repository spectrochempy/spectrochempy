"""
Characterization tests for integration semantics on NDDataset.

This suite characterizes CURRENT behavior of ``trapezoid()`` and
``simpson()``. It does NOT validate a desired future policy.

Coverage:
    - Return type, dimensionality reduction, default dim behavior
    - CoordSet reduction and surviving-coordinate preservation
    - Unit transformation semantics
    - Metadata propagation and operation-specific overrides
    - History / provenance behavior
    - Identity observations
    - Labels and masks
    - Edge cases: decreasing coords, non-uniform spacing, singleton inputs,
      unsupported all-dims / multi-axis integration, missing coordinates

Key observed patterns:
    - Reduction-like assembly over exactly one dimension
    - Result always remains an NDDataset, including 0-d results
    - Integration produces a derived scientific quantity
    - Title/description/history rewritten for that derived quantity
    - Name / author / origin / filename / meta preserved by copy-first assembly
    - Units combine data units with the integrated coordinate units
    - Mask information survives on the returned object, but masked source
      values still contribute numerically to the computed integral
"""

from pathlib import Path

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset


# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def integration_dataset():
    """Semantic-rich 2D dataset for integration characterization."""
    y = Coord(np.array([10.0, 20.0]), title="temperature", units="K")
    x = Coord(np.array([4.0, 2.0, 0.0]), title="wavenumber", units="cm^-1")
    ds = NDDataset(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        coordset=[y, x],
        units="absorbance",
        title="source_title",
        name="source_name",
    )
    ds.author = "test_author"
    ds.description = "source description"
    ds.origin = "test_origin"
    ds.filename = Path("source.spc")
    ds.meta.project = "test_project"
    ds.meta.tags = ["source"]
    ds.history = ["original history entry"]
    return ds


@pytest.fixture
def integration_1d():
    """1D dataset with decreasing coordinate for sign characterization."""
    x = Coord(np.array([2.0, 1.0, 0.0]), title="time", units="s")
    ds = NDDataset(np.array([1.0, 1.0, 1.0]), coordset=[x], units="V")
    ds.history = ["original history"]
    return ds


@pytest.fixture
def labeled_dataset():
    """2D dataset with labels on the surviving dimension."""
    y = Coord(np.array([0.0, 1.0, 2.0]), title="sample", labels=["a", "b", "c"])
    x = Coord(np.array([0.0, 1.0, 2.0]), title="x")
    return NDDataset(np.arange(9.0).reshape(3, 3), coordset=[y, x])


# ======================================================================================
# RETURN TYPE, SHAPE, DIMS
# ======================================================================================


class TestReturnTypeShapeDims:
    """Characterize return type, shape, and dimensionality reduction."""

    def test_trapezoid_returns_nddataset(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert isinstance(r, NDDataset)

    def test_simpson_returns_nddataset(self, integration_dataset):
        r = integration_dataset.simpson(dim="x")
        assert isinstance(r, NDDataset)

    def test_default_dim_is_x(self, integration_dataset):
        default = integration_dataset.trapezoid()
        explicit = integration_dataset.trapezoid(dim="x")
        np.testing.assert_allclose(default.data, explicit.data)
        assert default.dims == explicit.dims

    def test_integration_reduces_one_dimension(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.shape == (2,)
        assert r.dims == ["y"]

    def test_integrating_other_dim_keeps_remaining_dim(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="y")
        assert r.shape == (3,)
        assert r.dims == ["x"]

    def test_singleton_input_returns_zero_dim_nddataset(self):
        x = Coord(np.array([0.0]), title="x", units="cm^-1")
        ds = NDDataset(np.array([5.0]), coordset=[x], units="V")
        r = ds.trapezoid()
        assert isinstance(r, NDDataset)
        assert r.shape == ()
        assert r.dims == []

    def test_dim_none_is_not_supported(self, integration_dataset):
        with pytest.raises(TypeError, match="unexpected keyword argument 'dim'"):
            integration_dataset.trapezoid(dim=None)

    def test_keepdims_is_not_supported(self, integration_dataset):
        with pytest.raises(TypeError, match="unexpected keyword argument 'dim'"):
            integration_dataset.trapezoid(dim=None, keepdims=True)

    def test_multi_axis_via_axis_kwarg_not_supported(self, integration_dataset):
        with pytest.raises(TypeError, match="multiple values for keyword argument 'axis'"):
            integration_dataset.trapezoid(axis=(0, 1))

    def test_multi_axis_via_dim_tuple_not_supported(self, integration_dataset):
        with pytest.raises(TypeError, match="list indices must be integers or slices"):
            integration_dataset.trapezoid(dim=("y", "x"))


# ======================================================================================
# COORDSET
# ======================================================================================


class TestCoordSet:
    """Characterize coordset reduction behavior."""

    def test_integrated_coord_removed(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.coordset is not None
        assert r.coordset.names == ["y"]

    def test_surviving_coord_preserved(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        np.testing.assert_allclose(r.y.data, integration_dataset.y.data)
        assert r.y.title == integration_dataset.y.title
        assert r.y.units == integration_dataset.y.units

    def test_integrating_y_preserves_x_coord(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="y")
        np.testing.assert_allclose(r.x.data, integration_dataset.x.data)
        assert r.x.title == integration_dataset.x.title
        assert r.x.units == integration_dataset.x.units

    def test_zero_dim_result_has_empty_coordset(self, integration_1d):
        r = integration_1d.trapezoid()
        assert r.coordset is not None
        assert r.coordset.names == []


# ======================================================================================
# UNIT SEMANTICS
# ======================================================================================


class TestUnits:
    """Characterize unit transformation during integration."""

    def test_trapezoid_multiplies_data_and_coord_units(self, integration_1d):
        r = integration_1d.trapezoid()
        assert r.units == integration_1d.units * integration_1d.x.units

    def test_simpson_multiplies_data_and_coord_units(self, integration_1d):
        r = integration_1d.simpson()
        assert r.units == integration_1d.units * integration_1d.x.units

    def test_nonintegrated_coord_units_preserved(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.y.units == integration_dataset.y.units

    def test_unitless_coord_keeps_data_units(self):
        ds = NDDataset(
            np.array([1.0, 1.0, 1.0]),
            coordset=[Coord(np.array([0.0, 1.0, 2.0]), title="index")],
            units="V",
        )
        assert ds.trapezoid().units == ds.units

    def test_unitless_data_keeps_coord_units(self):
        ds = NDDataset(
            np.array([1.0, 1.0, 1.0]),
            coordset=[Coord(np.array([0.0, 1.0, 2.0]), title="time", units="s")],
        )
        assert ds.trapezoid().units == ds.x.units

    def test_no_units_stays_unitless(self):
        ds = NDDataset(
            np.array([1.0, 1.0, 1.0]),
            coordset=[Coord(np.array([0.0, 1.0, 2.0]), title="index")],
        )
        assert ds.trapezoid().units is None

    def test_decreasing_coord_produces_negative_area(self, integration_1d):
        r = integration_1d.trapezoid()
        assert np.isclose(r.data, -2.0)

    def test_nonuniform_spacing_participates_in_numeric_result(self):
        ds = NDDataset(
            np.array([0.0, 1.0, 1.0]),
            coordset=[Coord(np.array([0.0, 1.0, 3.0]), title="x", units="s")],
            units="V",
        )
        assert np.isclose(ds.trapezoid().data, 2.5)
        assert np.isclose(ds.simpson().data, 3.0)


# ======================================================================================
# METADATA
# ======================================================================================


class TestMetadata:
    """Characterize metadata propagation and operation-specific overrides."""

    def test_name_preserved(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.name == integration_dataset.name

    def test_title_overridden_to_area(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.title == "area"

    def test_description_rewritten(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.description == "Integration of NDDataset 'source_name' along dim: 'x'."

    def test_author_preserved(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.author == integration_dataset.author

    def test_origin_preserved(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.origin == integration_dataset.origin

    def test_filename_preserved(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.filename == integration_dataset.filename

    def test_meta_deepcopied(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.meta == integration_dataset.meta
        assert r.meta is not integration_dataset.meta
        assert r.meta.tags is not integration_dataset.meta.tags


# ======================================================================================
# HISTORY / PROVENANCE
# ======================================================================================


class TestHistoryProvenance:
    """Characterize history behavior after integration."""

    def test_history_replaced_not_appended(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert len(r.history) == 1
        assert "original history entry" not in r.history[0].lower()

    def test_history_mentions_method_name(self, integration_dataset):
        r = integration_dataset.simpson(dim="x")
        assert "`simpson` method" in r.history[0]

    def test_history_contains_timestamp_prefix(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert "> Dataset resulting from application of `trapezoid` method" in r.history[0]


# ======================================================================================
# IDENTITY
# ======================================================================================


class TestIdentity:
    """Characterize integration as a derived-quantity operation."""

    def test_integration_rewrites_identity_markers_for_derived_quantity(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert r.title == "area"
        assert r.description == "Integration of NDDataset 'source_name' along dim: 'x'."
        assert len(r.history) == 1
        assert "`trapezoid` method" in r.history[0]


# ======================================================================================
# LABELS AND MASKS
# ======================================================================================


class TestLabelsAndMasks:
    """Characterize label preservation and mask handling."""

    def test_labels_on_surviving_dimension_preserved(self, labeled_dataset):
        r = labeled_dataset.trapezoid(dim="x")
        assert r.y.is_labeled
        assert list(np.asarray(r.y.get_labels())) == ["a", "b", "c"]

    def test_result_mask_is_scalar_false_for_unmasked_input(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x")
        assert isinstance(r.mask, np.bool_)
        assert not bool(r.mask)
        assert not r.is_masked

    def test_masked_values_affect_mask_but_not_numeric_integration(self):
        arr = np.ma.MaskedArray([1.0, 200.0, 3.0], mask=[0, 1, 0])
        ds = NDDataset(
            arr,
            coordset=[Coord(np.array([0.0, 1.0, 2.0]), title="x", units="s")],
            units="V",
        )

        trap = ds.trapezoid()
        simp = ds.simpson()

        assert np.isclose(trap.data, 202.0)
        assert np.isclose(simp.data, 268.0)
        assert np.array_equal(trap.mask, [False, True, False])
        assert np.array_equal(simp.mask, [False, True, False])
        assert trap.is_masked
        assert simp.is_masked


# ======================================================================================
# EDGE CASES
# ======================================================================================


class TestEdgeCases:
    """Characterize selected edge-case behavior."""

    def test_no_coordset_raises_attribute_error(self):
        ds = NDDataset(np.array([1.0, 2.0, 3.0]), units="V")
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'data'"):
            ds.trapezoid()

    def test_partial_then_full_integration_stays_nddataset(self, integration_dataset):
        r = integration_dataset.trapezoid(dim="x").trapezoid(dim="y")
        assert isinstance(r, NDDataset)
        assert r.shape == ()
        assert r.dims == []
