"""
Characterization tests for indexing and selection semantics on NDDataset.

This suite characterizes CURRENT behavior of ds[...] indexing.
It does NOT validate a desired future policy.

Coverage:
    - Return type, shape, dims
    - CoordSet behavior (preserve/reduce/slice)
    - Units, masks, metadata, history
    - ROI, modeldata, labels
    - Identity, provenance
    - Integer, slice, ellipsis, step, label, float, fancy indexing
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset


# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def ds():
    """Semantic-rich 2D dataset with labels on x.

    - dims: ['y', 'x'] (5, 7)
    - CoordSet with titles, units, labels
    - Full metadata
    """
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(
        np.linspace(4000.0, 1000.0, 7),
        title="wavenumber",
        units="cm^-1",
        labels=["a", "b", "c", "d", "e", "f", "g"],
    )
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
    ds.modeldata = np.full((5, 7), 42.0)
    ds.history = ["original entry"]
    return ds


@pytest.fixture
def ds_unitful():
    """2D dataset with data units."""
    x = Coord(np.linspace(4000.0, 1000.0, 7))
    y = Coord(np.linspace(0.0, 60.0, 5))
    return NDDataset(
        np.arange(35.0).reshape(5, 7), coordset=[y, x], units="m"
    )


@pytest.fixture
def ds_masked():
    """2D dataset with a masked value at [0, 0]."""
    arr = np.ma.MaskedArray(
        np.arange(35.0).reshape(5, 7), mask=np.zeros((5, 7), dtype=bool)
    )
    arr[0, 0] = np.ma.masked
    return NDDataset(arr)


@pytest.fixture
def ds_1d():
    """Simple 1D dataset."""
    return NDDataset(np.array([10.0, 20.0, 30.0]))


# ======================================================================================
# RETURN TYPE
# ======================================================================================


class TestReturnType:
    """Characterize return type for various indexing forms.

    Observation: all indexing forms that return a result return NDDataset.
    Scalar extraction (ds[0, 0]) returns NDDataset with singleton dims.
    """

    def test_single_index_returns_nddataset(self, ds):
        r = ds[0]
        assert isinstance(r, NDDataset)

    def test_column_slice_returns_nddataset(self, ds):
        r = ds[:, 0]
        assert isinstance(r, NDDataset)

    def test_row_range_returns_nddataset(self, ds):
        r = ds[1:3]
        assert isinstance(r, NDDataset)

    def test_scalar_like_returns_nddataset(self, ds):
        r = ds[0, 0]
        assert isinstance(r, NDDataset)

    def test_ellipsis_returns_nddataset(self, ds):
        r = ds[..., 0]
        assert isinstance(r, NDDataset)

    def test_negative_index_returns_nddataset(self, ds):
        r = ds[-1]
        assert isinstance(r, NDDataset)

    def test_step_slice_returns_nddataset(self, ds):
        r = ds[:, ::2]
        assert isinstance(r, NDDataset)

    def test_label_slice_returns_nddataset(self, ds):
        r = ds[:, "a":"d"]
        assert isinstance(r, NDDataset)

    def test_label_index_returns_nddataset(self, ds):
        r = ds[:, "c"]
        assert isinstance(r, NDDataset)

    def test_bool_fancy_returns_nddataset(self, ds):
        mask = np.array([True, False, True, False, True])
        r = ds[mask]
        assert isinstance(r, NDDataset)

    def test_list_fancy_returns_nddataset(self, ds):
        r = ds[[0, 2, 4]]
        assert isinstance(r, NDDataset)

    def test_float_index_returns_nddataset(self, ds):
        r = ds[2000.0]
        assert isinstance(r, NDDataset)

    def test_string_coord_access_returns_coord(self, ds):
        r = ds["x"]
        assert isinstance(r, Coord)


# ======================================================================================
# SHAPE AND DIMS
# ======================================================================================


class TestShapeAndDims:
    """Characterize shape and dims after indexing."""

    def test_single_index_preserves_dims_with_singleton(self, ds):
        r = ds[0]
        assert r.shape == (1, 7)
        assert r.dims == ["y", "x"]

    def test_last_index_preserves_dims_with_singleton(self, ds):
        r = ds[-1]
        assert r.shape == (1, 7)
        assert r.dims == ["y", "x"]

    def test_column_selection_preserves_dims_with_singleton(self, ds):
        r = ds[:, 0]
        assert r.shape == (5, 1)
        assert r.dims == ["y", "x"]

    def test_row_range(self, ds):
        r = ds[1:3]
        assert r.shape == (2, 7)
        assert r.dims == ["y", "x"]

    def test_column_range(self, ds):
        r = ds[:, 2:5]
        assert r.shape == (5, 3)
        assert r.dims == ["y", "x"]

    def test_submatrix(self, ds):
        r = ds[1:4, 2:5]
        assert r.shape == (3, 3)
        assert r.dims == ["y", "x"]

    def test_scalar_like_singleton(self, ds):
        r = ds[0, 0]
        assert r.shape == (1, 1)
        assert r.dims == ["y", "x"]

    def test_ellipsis_first(self, ds):
        r = ds[..., 0]
        assert r.shape == (5, 1)
        assert r.dims == ["y", "x"]

    def test_ellipsis_last(self, ds):
        r = ds[0, ...]
        assert r.shape == (1, 7)
        assert r.dims == ["y", "x"]

    def test_step_slice_y(self, ds):
        r = ds[::2, :]
        assert r.shape == (3, 7)
        assert r.dims == ["y", "x"]

    def test_step_slice_x(self, ds):
        r = ds[:, ::2]
        assert r.shape == (5, 4)
        assert r.dims == ["y", "x"]

    def test_negative_step_y(self, ds):
        r = ds[::-1, :]
        assert r.shape == (5, 7)
        assert r.dims == ["y", "x"]

    def test_bool_fancy_shape(self, ds):
        mask = np.array([True, False, True, False, True])
        r = ds[mask]
        assert r.shape == (3, 7)
        assert r.dims == ["y", "x"]

    def test_list_fancy_shape(self, ds):
        r = ds[[0, 2, 4]]
        assert r.shape == (3, 7)
        assert r.dims == ["y", "x"]

    def test_1d_single_index(self, ds_1d):
        r = ds_1d[0]
        assert r.shape == (1,)

    def test_label_slice_shape(self, ds):
        r = ds[:, "a":"d"]
        assert r.shape == (5, 4)
        assert r.dims == ["y", "x"]

    def test_label_index_shape(self, ds):
        r = ds[:, "c"]
        assert r.shape == (5, 1)
        assert r.dims == ["y", "x"]


# ======================================================================================
# COORDSET BEHAVIOR
# ======================================================================================


class TestCoordSetBehavior:
    """Characterize CoordSet propagation through indexing."""

    def test_single_index_coord_sliced(self, ds):
        r = ds[0]
        assert r.y.size == 1

    def test_single_index_coord_preserved(self, ds):
        r = ds[0]
        assert r.x.size == 7

    def test_column_index_coord_sliced(self, ds):
        r = ds[:, 0]
        assert r.x.size == 1

    def test_column_index_other_coord_preserved(self, ds):
        r = ds[:, 0]
        assert r.y.size == 5

    def test_row_range_coord_sliced(self, ds):
        r = ds[1:3]
        assert r.y.size == 2

    def test_submatrix_both_coords_sliced(self, ds):
        r = ds[1:4, 2:5]
        assert r.y.size == 3
        assert r.x.size == 3

    def test_coord_titles_preserved(self, ds):
        r = ds[0]
        assert r.y.title == "time"
        assert r.x.title == "wavenumber"

    def test_coord_units_preserved(self, ds):
        r = ds[0]
        assert str(r.y.units) == "s"
        assert str(r.x.units) == "cm⁻¹"

    def test_coordset_names_preserved(self, ds):
        r = ds[0]
        assert "y" in r.coordset.names
        assert "x" in r.coordset.names

    def test_step_slice_coord_sliced(self, ds):
        r = ds[::2, :]
        expected_y = np.array([0.0, 30.0, 60.0])
        assert np.allclose(r.y.data, expected_y)

    def test_negative_step_coord_reversed(self, ds):
        r = ds[::-1, :]
        assert np.allclose(r.y.data, ds.y.data[::-1])

    def test_bool_fancy_coord_sliced(self, ds):
        mask = np.array([True, False, True, False, True])
        r = ds[mask]
        expected_y = np.array([0.0, 30.0, 60.0])
        assert np.allclose(r.y.data, expected_y)

    def test_list_fancy_coord_sliced(self, ds):
        r = ds[[0, 2, 4]]
        expected_y = np.array([0.0, 30.0, 60.0])
        assert np.allclose(r.y.data, expected_y)

    def test_ellipsis_slices_correct_dim(self, ds):
        r = ds[..., 0]
        assert r.x.size == 1
        assert r.y.size == 5


# ======================================================================================
# UNITS
# ======================================================================================


class TestUnits:
    """Characterize unit behavior after indexing."""

    def test_units_preserved_on_single_index(self, ds_unitful):
        r = ds_unitful[0]
        assert str(r.units) == "m"

    def test_units_preserved_on_scalar_like(self, ds_unitful):
        r = ds_unitful[0, 0]
        assert str(r.units) == "m"

    def test_units_preserved_on_slice(self, ds_unitful):
        r = ds_unitful[1:3, :]
        assert str(r.units) == "m"

    def test_units_preserved_on_bool_fancy(self, ds_unitful):
        mask = np.array([True, False, True, False, True])
        r = ds_unitful[mask]
        assert str(r.units) == "m"

    def test_units_preserved_on_list_fancy(self, ds_unitful):
        r = ds_unitful[[0, 2, 4]]
        assert str(r.units) == "m"


# ======================================================================================
# MASKS
# ======================================================================================


class TestMasks:
    """Characterize mask behavior through indexing."""

    def test_mask_sliced_with_data(self, ds_masked):
        r = ds_masked[0]
        assert r.mask[0, 0]

    def test_mask_on_column_index(self, ds_masked):
        r = ds_masked[:, 0]
        assert r.mask[0, 0]

    def test_scalar_like_mask_preserved(self, ds_masked):
        r = ds_masked[0, 0]
        assert r.mask[0, 0]

    def test_mask_absent_after_slice_without_masked(self, ds_masked):
        r = ds_masked[1:3, 2:5]
        assert not np.any(r.mask)

    def test_unmasked_slice_mask_is_false(self, ds_masked):
        """Notable: slices that don't include masked elements
        may have a scalar False mask rather than an array."""
        r = ds_masked[2:5, 2:5]
        assert not np.any(r.mask)

    def test_bool_fancy_mask_preserved(self, ds_masked):
        mask = np.array([True, False, True, False, True])
        r = ds_masked[mask]
        assert r.mask[0, 0]


# ======================================================================================
# METADATA
# ======================================================================================


class TestMetadata:
    """Characterize metadata propagation through indexing.

    Observations: indexing uses copy-first assembly. All metadata
    (title, name, author, description, origin, custom meta) is
    preserved through copying.
    """

    def test_title_preserved(self, ds):
        r = ds[0]
        assert r.title == "ds_title"

    def test_name_preserved(self, ds):
        r = ds[0]
        assert r.name == "ds_name"

    def test_author_preserved(self, ds):
        r = ds[0]
        assert r.author == "test_author"

    def test_description_preserved(self, ds):
        r = ds[0]
        assert r.description == "test description"

    def test_origin_preserved(self, ds):
        r = ds[0]
        assert r.origin == "test_origin"

    def test_meta_project_preserved(self, ds):
        r = ds[0]
        assert r.meta.project == "test_project"

    def test_meta_deep_copied(self, ds):
        r = ds[0]
        r.meta.project = "modified"
        assert ds.meta.project == "test_project"

    def test_title_preserved_on_slice(self, ds):
        r = ds[1:4, 2:5]
        assert r.title == "ds_title"

    def test_title_preserved_on_scalar_like(self, ds):
        r = ds[0, 0]
        assert r.title == "ds_title"

    def test_title_preserved_on_fancy(self, ds):
        r = ds[[0, 2, 4]]
        assert r.title == "ds_title"


# ======================================================================================
# HISTORY
# ======================================================================================


class TestHistory:
    """Characterize history behavior through indexing.

    Observations: history is APPENDED for all indexing forms.
    Original entries are preserved, and a new entry is added
    describing the slice operation.
    """

    def test_history_appended_on_single_index(self, ds):
        r = ds[0]
        assert len(r.history) == 2
        assert "Original entry" in r.history[0]
        assert "Slice extracted" in r.history[1]

    def test_history_appended_on_slice(self, ds):
        r = ds[1:3]
        assert len(r.history) == 2
        assert "Slice extracted" in r.history[1]

    def test_history_appended_on_double_slice(self, ds):
        r = ds[1:4, 2:5]
        assert len(r.history) == 2

    def test_history_appended_on_scalar_like(self, ds):
        r = ds[0, 0]
        assert len(r.history) == 2

    def test_history_appended_on_fancy_bool(self, ds):
        mask = np.array([True, False, True, False, True])
        r = ds[mask]
        assert len(r.history) == 2

    def test_history_appended_on_fancy_list(self, ds):
        r = ds[[0, 2, 4]]
        assert len(r.history) == 2

    def test_history_appended_on_label_slice(self, ds):
        r = ds[:, "a":"d"]
        assert len(r.history) == 2

    def test_history_appended_on_step_slice(self, ds):
        r = ds[::2, :]
        assert len(r.history) == 2

    def test_original_history_preserved(self, ds):
        r = ds[0]
        assert "Original entry" in r.history[0]


# ======================================================================================
# ROI
# ======================================================================================


class TestRoi:
    """Characterize ROI behavior through indexing.

    Observations: roi is copied unchanged through all indexing forms.
    It is NOT sliced or adjusted to match the new data shape.
    """

    def test_roi_preserved_on_single_index(self, ds):
        r = ds[0]
        assert r.roi == [0.0, 10.0]

    def test_roi_preserved_on_slice(self, ds):
        r = ds[1:3]
        assert r.roi == [0.0, 10.0]

    def test_roi_preserved_on_submatrix(self, ds):
        r = ds[1:4, 2:5]
        assert r.roi == [0.0, 10.0]

    def test_roi_preserved_on_scalar_like(self, ds):
        r = ds[0, 0]
        assert r.roi == [0.0, 10.0]

    def test_roi_preserved_on_fancy(self, ds):
        r = ds[[0, 2, 4]]
        assert r.roi == [0.0, 10.0]


# ======================================================================================
# MODELDATA
# ======================================================================================


class TestModeldata:
    """Characterize modeldata behavior through indexing.

    Notable behavior: modeldata is NOT sliced with the data.
    It retains the original full shape even after subsetting.
    """

    def test_modeldata_stale_after_single_index(self, ds):
        r = ds[0]
        assert r.modeldata.shape == (5, 7)

    def test_modeldata_stale_after_slice(self, ds):
        r = ds[1:3]
        assert r.modeldata.shape == (5, 7)

    def test_modeldata_stale_after_submatrix(self, ds):
        r = ds[1:4, 2:5]
        assert r.modeldata.shape == (5, 7)

    def test_modeldata_stale_after_scalar_like(self, ds):
        r = ds[0, 0]
        assert r.modeldata.shape == (5, 7)

    def test_modeldata_stale_after_fancy(self, ds):
        r = ds[[0, 2, 4]]
        assert r.modeldata.shape == (5, 7)

    def test_modeldata_values_preserved(self, ds):
        r = ds[0]
        assert np.allclose(r.modeldata, 42.0)


# ======================================================================================
# LABELS
# ======================================================================================


class TestLabels:
    """Characterize label behavior through indexing."""

    def test_labels_sliced_on_column_slice(self, ds):
        r = ds[:, 2:5]
        expected = ["c", "d", "e"]
        assert list(r.x.labels) == expected

    def test_labels_on_single_column_index(self, ds):
        r = ds[:, 0]
        assert list(r.x.labels) == ["a"]

    def test_labels_on_label_slice(self, ds):
        r = ds[:, "a":"d"]
        assert list(r.x.labels) == ["a", "b", "c", "d"]

    def test_labels_on_label_index(self, ds):
        r = ds[:, "c"]
        assert list(r.x.labels) == ["c"]

    def test_labels_preserved_on_step_slice(self, ds):
        r = ds[:, ::2]
        assert list(r.x.labels) == ["a", "c", "e", "g"]

    def test_labels_on_negative_step(self, ds):
        r = ds[:, ::-1]
        expected = ["g", "f", "e", "d", "c", "b", "a"]
        assert list(r.x.labels) == expected

    def test_labels_on_single_row(self, ds):
        r = ds[0]
        assert r.y.labels is None or len(r.y.labels) == 1


# ======================================================================================
# IDENTITY
# ======================================================================================


class TestIdentity:
    """Characterize scientific object identity after indexing.

    Observations:
    - Slicing preserves identity: title, name, author, description,
      origin, meta all survive via copy-first assembly.
    - Even scalar-like extraction (ds[0,0]) returns NDDataset with
      full identity — identity never leaves the dataset surface.
    """

    def test_slicing_identity_preserved(self, ds):
        r = ds[1:4, 2:5]
        assert r.title == ds.title
        assert r.name == ds.name
        assert r.author == ds.author
        assert r.description == ds.description
        assert r.origin == ds.origin

    def test_scalar_like_identity_preserved(self, ds):
        r = ds[0, 0]
        assert r.title == ds.title
        assert r.name == ds.name

    def test_coordset_identity_support_sliced(self, ds):
        r = ds[1:4, 2:5]
        assert r.y.size == 3
        assert r.x.size == 3

    def test_identity_no_new_history_overwrite(self, ds):
        """Notable: original history is preserved, not replaced."""
        r = ds[0]
        assert "Original entry" in r.history[0]


# ======================================================================================
# PROVENANCE
# ======================================================================================


class TestProvenance:
    """Characterize provenance through indexing.

    Observations:
    - origin and author are preserved via copy-first.
    - history is extended (appended), not rewritten.
    - Provenance is preserved for all indexing forms.
    """

    def test_origin_preserved(self, ds):
        r = ds[0]
        assert r.origin == "test_origin"

    def test_author_preserved(self, ds):
        r = ds[0]
        assert r.author == "test_author"

    def test_origin_preserved_after_scalar_like(self, ds):
        r = ds[0, 0]
        assert r.origin == "test_origin"

    def test_history_extended_not_rewritten(self, ds):
        r = ds[0]
        assert len(r.history) > 1
        assert "Original entry" in r.history[0]

    def test_provenance_for_label_slice(self, ds):
        r = ds[:, "a":"d"]
        assert r.origin == ds.origin
        assert r.author == ds.author

    def test_provenance_for_fancy(self, ds):
        r = ds[[0, 2, 4]]
        assert r.origin == ds.origin
        assert r.author == ds.author


# ======================================================================================
# SCALAR EXTRACTION
# ======================================================================================


class TestScalarExtraction:
    """Characterize behavior of single-element extraction.

    Notable behavior: ds[0, 0] returns NDDataset(1,1), not a scalar.
    All identity and provenance is preserved. The object never leaves
    the NDDataset surface.
    """

    def test_scalar_like_value(self, ds):
        r = ds[0, 0]
        assert r.data[0, 0] == 0.0

    def test_scalar_like_different_value(self, ds):
        r = ds[2, 3]
        assert r.data[0, 0] == 17.0

    def test_scalar_like_is_nddataset(self, ds):
        r = ds[0, 0]
        assert isinstance(r, NDDataset)

    def test_scalar_like_has_dims(self, ds):
        r = ds[0, 0]
        assert len(r.dims) == 2

    def test_scalar_like_singleton_dims(self, ds):
        r = ds[0, 0]
        assert r.shape == (1, 1)

    def test_scalar_like_history(self, ds):
        r = ds[0, 0]
        assert len(r.history) == 2


# ======================================================================================
# LABEL-BASED INDEXING
# ======================================================================================


class TestLabelIndexing:
    """Characterize label-based indexing behavior."""

    def test_label_slice_inclusive(self, ds):
        """Notable: label slices are inclusive on both ends."""
        r = ds[:, "a":"d"]
        assert list(r.x.labels) == ["a", "b", "c", "d"]

    def test_label_index_singleton(self, ds):
        r = ds[:, "c"]
        assert list(r.x.labels) == ["c"]
        assert r.x.size == 1

    def test_label_slice_with_step(self, ds):
        r = ds[:, "a":"g":2]
        assert list(r.x.labels) == ["a", "c", "e", "g"]

    def test_label_slice_reversed_gives_original_order(self, ds):
        """Notable: reversed label slice returns labels in original
        order (a..g), not reversed. A negative step returns None."""
        r = ds[:, "g":"a"]
        assert list(r.x.labels) == ["a", "b", "c", "d", "e", "f", "g"]

    def test_label_on_y_without_labels(self, ds):
        """Notable: slicing a dim without labels by string raises
        IndexError because the string is not found as a label."""
        with pytest.raises(IndexError):
            ds["a":"c", :]


# ======================================================================================
# ELLIPSIS INDEXING
# ======================================================================================


class TestEllipsis:
    """Characterize ellipsis (...) behavior."""

    def test_ellipsis_first_pos(self, ds):
        r = ds[..., 0]
        assert r.shape == (5, 1)

    def test_ellipsis_last_pos(self, ds):
        r = ds[0, ...]
        assert r.shape == (1, 7)

    def test_ellipsis_alone(self, ds):
        r = ds[...]
        assert r.shape == (5, 7)

    def test_ellipsis_coord_slices_correctly(self, ds):
        r = ds[..., 0]
        assert r.x.size == 1
        assert r.y.size == 5


# ======================================================================================
# SLICE WITH STEP
# ======================================================================================


class TestStepSlice:
    """Characterize step slice behavior."""

    def test_step_2_y(self, ds):
        r = ds[::2, :]
        assert r.shape == (3, 7)
        assert np.allclose(r.y.data, [0.0, 30.0, 60.0])

    def test_step_2_x(self, ds):
        r = ds[:, ::2]
        assert r.shape == (5, 4)
        assert np.allclose(r.x.data, [4000.0, 3000.0, 2000.0, 1000.0])

    def test_negative_step_y(self, ds):
        r = ds[::-1, :]
        assert np.allclose(r.y.data, [60.0, 45.0, 30.0, 15.0, 0.0])

    def test_negative_step_x(self, ds):
        r = ds[:, ::-1]
        expected = np.linspace(1000.0, 4000.0, 7)
        assert np.allclose(r.x.data, expected)


# ======================================================================================
# FLOAT INDEXING
# ======================================================================================


class TestFloatIndexing:
    """Characterize float-based nearest-value indexing."""

    def test_float_closest(self, ds):
        r = ds[2000.0]
        assert r.shape == (1, 7)

    def test_float_on_x(self, ds):
        r = ds[:, 2500.0]
        assert r.shape == (5, 1)

    def test_float_out_of_limits_clips(self, ds):
        """Notable: out-of-limits float returns nearest boundary value
        with a log message, not None."""
        r = ds[5000.0]
        assert r is not None
        assert isinstance(r, NDDataset)


# ======================================================================================
# DATA CORRECTNESS
# ======================================================================================


class TestDataCorrectness:
    """Characterize data value correctness."""

    def test_single_index_value(self, ds):
        r = ds[0]
        assert np.array_equal(r.data[0], ds.data[0])

    def test_column_index_value(self, ds):
        r = ds[:, 0]
        assert np.array_equal(r.data[:, 0], ds.data[:, 0])

    def test_submatrix_values(self, ds):
        r = ds[1:4, 2:5]
        assert np.array_equal(r.data, ds.data[1:4, 2:5])

    def test_scalar_like_value(self, ds):
        r = ds[0, 0]
        assert r.data[0, 0] == ds.data[0, 0]


# ======================================================================================
# 1D DATASET
# ======================================================================================


class Test1DDataset:
    """Characterize indexing on 1D datasets."""

    def test_1d_single_index(self, ds_1d):
        r = ds_1d[0]
        assert r.shape == (1,)

    def test_1d_range(self, ds_1d):
        r = ds_1d[1:3]
        assert r.shape == (2,)

    def test_1d_step(self, ds_1d):
        r = ds_1d[::2]
        assert r.shape == (2,)


# ======================================================================================
# EDGE CASES
# ======================================================================================


class TestEdgeCases:
    """Characterize error handling and edge cases."""

    def test_too_many_indices_raises(self, ds):
        with pytest.raises(IndexError):
            ds[0, 0, 0]

    def test_integer_out_of_range_returns_none(self, ds):
        """Notable: out-of-range integer returns None (with error log),
        rather than raising IndexError directly."""
        r = ds[100]
        assert r is None

    def test_string_coord_access(self, ds):
        r = ds["x"]
        assert isinstance(r, Coord)

    def test_double_slice_metadata(self, ds):
        r = ds[1:4, 2:5]
        assert r.title == ds.title
        assert r.name == ds.name
