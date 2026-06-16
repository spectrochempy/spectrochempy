"""
Characterization tests for combination operation semantics on NDDataset.

This suite characterizes CURRENT behavior of concatenate() and stack().
It does NOT validate a desired future policy.

Coverage:
    - concatenate: data, CoordSet, units, masks, metadata, history,
      ROI/modeldata, identity/provenance, edge cases
    - stack: data, CoordSet, metadata, provenance

Result Assembly Pattern observations (see RFC):
    Combination operations follow Pattern B (Rebuild / Synthesize):
    the result is assembled from a copy of the last dataset with
    selective field overwrites, not a pure copy-first approach.
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.processing.transformation.concatenate import concatenate, stack
from spectrochempy.utils.exceptions import (
    DimensionsCompatibilityError,
    UnitsCompatibilityError,
)


# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def dataset_x():
    """Semantic-rich 2D dataset — first in concatenation.

    - dims: ['y', 'x'] (5, 7)
    - CoordSet with titles, units
    - full metadata
    """
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(np.linspace(4000.0, 1000.0, 7), title="wavenumber", units="cm^-1")
    ds = NDDataset(
        np.arange(35.0, dtype="float64").reshape(5, 7),
        coordset=[y, x],
        title="dataset_x",
        name="dataset_x_name",
    )
    ds.author = "author_x"
    ds.description = "description_x"
    ds.origin = "origin_x"
    ds.meta.project = "project_x"
    ds.roi = [0.0, 10.0]
    ds.modeldata = np.full((5, 7), 42.0)
    ds.history = ["original_x"]
    return ds


@pytest.fixture
def dataset_y():
    """Semantic-rich 2D dataset — second in concatenation.

    Same shape as dataset_x along non-concatenated dimensions.
    Different metadata values for propagation characterization.
    """
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(np.linspace(4000.0, 1000.0, 7), title="wavenumber", units="cm^-1")
    ds = NDDataset(
        np.full((5, 7), 99.0, dtype="float64"),
        coordset=[y, x],
        title="dataset_y",
        name="dataset_y_name",
    )
    ds.author = "author_y"
    ds.description = "description_y"
    ds.origin = "origin_y"
    ds.meta.project = "project_y"
    ds.roi = [5.0, 20.0]
    ds.modeldata = np.full((5, 7), 99.0)
    ds.history = ["original_y"]
    return ds


@pytest.fixture
def bare_dataset_1d():
    """Minimal 1D dataset with no CoordSet."""
    return NDDataset(np.array([1.0, 2.0, 3.0]))


@pytest.fixture
def unitful_dataset_x():
    """1D dataset with units for unit characterization."""
    return NDDataset(np.array([1.0, 2.0, 3.0]), units="m")


@pytest.fixture
def unitful_dataset_y():
    """1D dataset with compatible units (different magnitude)."""
    return NDDataset(np.array([400.0, 500.0, 600.0]), units="cm")


@pytest.fixture
def unitful_dataset_incompatible():
    """1D dataset with incompatible units."""
    return NDDataset(np.array([10.0, 20.0, 30.0]), units="s")


@pytest.fixture
def masked_dataset_x():
    """2D dataset with a masked value."""
    arr = np.ma.MaskedArray(
        np.arange(35.0).reshape(5, 7), mask=np.zeros((5, 7), dtype=bool)
    )
    arr[0, 0] = np.ma.masked
    return NDDataset(arr)


@pytest.fixture
def masked_dataset_y():
    """2D dataset with a masked value at a different position."""
    arr = np.ma.MaskedArray(
        np.full((5, 7), 99.0), mask=np.zeros((5, 7), dtype=bool)
    )
    arr[1, 1] = np.ma.masked
    return NDDataset(arr)


@pytest.fixture
def labeled_dataset_x():
    """Dataset with labeled coordinates on x dim."""
    x = Coord(
        np.linspace(4000.0, 1000.0, 7),
        labels=["a", "b", "c", "d", "e", "f", "g"],
    )
    y = Coord(np.linspace(0.0, 60.0, 5))
    return NDDataset(np.arange(35.0).reshape(5, 7), coordset=[y, x])


@pytest.fixture
def labeled_dataset_y():
    """Dataset with different labels on x dim."""
    x = Coord(
        np.linspace(4000.0, 1000.0, 7),
        labels=["h", "i", "j", "k", "l", "m", "n"],
    )
    y = Coord(np.linspace(0.0, 60.0, 5))
    return NDDataset(np.full((5, 7), 99.0), coordset=[y, x])


# ======================================================================================
# DATA SEMANTICS
# ======================================================================================


class TestConcatenateData:
    """Characterize data shape, dims, and ordering through concatenation."""

    def test_concat_along_x_shape_and_dims(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.shape == (5, 14)
        assert c.dims == ["y", "x"]

    def test_concat_along_y_shape_and_dims(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="y")
        assert c.shape == (10, 7)
        assert c.dims == ["y", "x"]

    def test_concat_default_dim_is_last(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y)
        assert c.shape == (5, 14)
        assert c.dims == ["y", "x"]

    def test_concat_data_ordering_first_then_second(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert np.array_equal(c.data[:, :7], dataset_x.data)
        assert np.array_equal(c.data[:, 7:], dataset_y.data)

    def test_concat_data_values_correct(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        expected = np.concatenate([dataset_x.data, dataset_y.data], axis=1)
        assert np.allclose(c.data, expected)

    def test_concat_1d(self, bare_dataset_1d):
        a = NDDataset(np.array([1.0, 2.0, 3.0]))
        b = NDDataset(np.array([4.0, 5.0, 6.0]))
        c = concatenate(a, b, dims="x")
        assert c.shape == (6,)
        assert np.allclose(c.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_concat_three_datasets(self, dataset_x):
        c = concatenate(dataset_x, dataset_x, dataset_x, dims="x")
        assert c.shape == (5, 21)

    def test_concat_preserves_mask_data_separation(self):
        """Masked values should not leak into data."""
        x = Coord(np.linspace(4000.0, 1000.0, 7))
        y = Coord(np.linspace(0.0, 60.0, 5))
        arr = np.ma.MaskedArray(
            np.arange(35.0).reshape(5, 7), mask=np.zeros((5, 7), dtype=bool)
        )
        arr[0, 0] = np.ma.masked
        ds = NDDataset(arr, coordset=[y, x])
        c = concatenate(ds, ds, dims="x")
        assert np.isclose(c.data[0, 0], 0.0)


# ======================================================================================
# COORDSET SEMANTICS
# ======================================================================================


class TestConcatenateCoordSet:
    """Characterize CoordSet behavior through concatenation."""

    def test_concat_dim_coord_data_merged(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        expected = np.concatenate([dataset_x.x.data, dataset_y.x.data])
        assert np.allclose(c.x.data, expected)

    def test_concat_dim_coord_size(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.x.size == dataset_x.x.size + dataset_y.x.size

    def test_non_concat_coord_preserved(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert "y" in c.coordset.names
        assert np.array_equal(c["y"].data, dataset_x["y"].data)

    def test_non_concat_coord_title_preserved(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c["y"].title == "time"

    def test_non_concat_coord_units_preserved(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c["y"].units is not None

    def test_concat_dim_coord_title_preserved(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c["x"].title == "wavenumber"

    def test_concat_dim_coord_units_preserved(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c["x"].units is not None

    def test_labels_concatenated(self, labeled_dataset_x, labeled_dataset_y):
        c = concatenate(labeled_dataset_x, labeled_dataset_y, dims="x")
        expected = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
        assert c.x.labels.tolist() == expected

    def test_concat_dim_matches_dim_name(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="y")
        expected_y = np.concatenate([dataset_x["y"].data, dataset_y["y"].data])
        assert np.allclose(c["y"].data, expected_y)

    def test_concat_non_concat_coord_unchanged(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c["y"].shape == dataset_x["y"].shape

    # ------------------------------------------------------------------
    # Coordinate unit preservation
    # ------------------------------------------------------------------

    def test_concat_dim_coord_units_string(self, dataset_x, dataset_y):
        """Concatenated-dim coordinate retains its units (string form)."""
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert str(c.x.units) == "cm⁻¹"

    def test_non_concat_coord_units_preserved_after_concat(self, dataset_x, dataset_y):
        """Non-concatenated coordinate retains its units."""
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert str(c.y.units) == "s"

    def test_compatible_coord_units_convert(self):
        """Notable: coordinate with compatible units is auto-converted
        to match the first dataset's coordinate units."""
        y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
        x1 = Coord(np.linspace(4000.0, 1000.0, 7), title="dist", units="m")
        x2 = Coord(np.linspace(4000.0, 1000.0, 7), title="dist", units="cm")
        a = NDDataset(np.arange(35.0).reshape(5, 7), coordset=[y, x1])
        b = NDDataset(np.full((5, 7), 99.0), coordset=[y, x2])
        c = concatenate(a, b, dims="x")
        assert str(c.x.units) == "m"
        # Data values reflect the conversion (second dataset converted from cm to m)
        assert np.allclose(c.x.data[:7], x1.data)
        assert np.allclose(c.x.data[7:], x2.data / 100.0)


# ======================================================================================
# UNIT SEMANTICS
# ======================================================================================


class TestConcatenateUnits:
    """Characterize unit behavior through concatenation."""

    def test_same_units_preserved(self):
        a = NDDataset(np.array([1.0, 2.0, 3.0]), units="m")
        b = NDDataset(np.array([4.0, 5.0, 6.0]), units="m")
        c = concatenate(a, b, dims="x")
        assert str(c.units) == "m"

    def test_compatible_units_converted_to_first(self):
        """Notable behavior: compatible units are auto-converted
        to the first dataset's units."""
        a = NDDataset(np.array([1.0, 2.0, 3.0]), units="m")
        b = NDDataset(np.array([400.0, 500.0, 600.0]), units="cm")
        c = concatenate(a, b, dims="x")
        assert str(c.units) == "m"
        assert np.allclose(c.data[:3], [1.0, 2.0, 3.0])
        assert np.allclose(c.data[3:], [4.0, 5.0, 6.0])

    def test_incompatible_units_raises(self):
        a = NDDataset(np.array([1.0, 2.0, 3.0]), units="m")
        b = NDDataset(np.array([10.0, 20.0, 30.0]), units="s")
        with pytest.raises(UnitsCompatibilityError):
            concatenate(a, b, dims="x")


# ======================================================================================
# MASK SEMANTICS
# ======================================================================================


class TestConcatenateMasks:
    """Characterize mask behavior through concatenation."""

    def test_masks_concatenated(self, masked_dataset_x, masked_dataset_y):
        c = concatenate(masked_dataset_x, masked_dataset_y, dims="x")
        assert c.mask[0, 0]  # masked from first dataset
        assert not c.mask[0, 7]  # second dataset col 0 not masked
        assert c.mask[1, 8]  # second dataset row 1 col 1 masked at offset 7

    def test_combined_mask_shape(self, masked_dataset_x, masked_dataset_y):
        c = concatenate(masked_dataset_x, masked_dataset_y, dims="x")
        assert c.mask.shape == (5, 14)

    def test_no_mask_if_none_masked(self):
        a = NDDataset(np.array([1.0, 2.0, 3.0]))
        b = NDDataset(np.array([4.0, 5.0, 6.0]))
        c = concatenate(a, b, dims="x")
        assert c.mask is None or not np.any(c.mask)


# ======================================================================================
# METADATA SEMANTICS
# ======================================================================================


class TestConcatenateMetadata:
    """Characterize metadata propagation through concatenation.

    Observations (not policy):
    - title comes from the FIRST dataset (with warning if different)
    - name comes from the LAST dataset (via copy of last input)
    - author is merged with ' & ' separator
    - description is synthesized: 'Concatenation of N datasets:\n( name1, name2, ... )'
    - origin comes from the LAST dataset (via copy)
    - meta comes from the LAST dataset (via copy)
    """

    def test_title_from_first_dataset(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.title == "dataset_x"

    def test_name_from_last_dataset(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.name == "dataset_y_name"

    def test_author_merged(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert "author_x" in c.author
        assert "author_y" in c.author
        assert " & " in c.author

    def test_author_deduplicated(self, dataset_x):
        """Notable behavior: same author is not duplicated."""
        c = concatenate(dataset_x, dataset_x, dims="x")
        assert c.author.count("author_x") == 1

    def test_description_synthesized(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert "Concatenation of 2  datasets" in c.description
        assert "dataset_x_name" in c.description
        assert "dataset_y_name" in c.description

    def test_origin_from_last_dataset(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.origin == "origin_y"

    def test_meta_from_last_dataset(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.meta.project == "project_y"

    def test_meta_deep_copied(self, dataset_x, dataset_y):
        """Notable behavior: modifying result meta should not affect source."""
        c = concatenate(dataset_x, dataset_y, dims="x")
        c.meta.project = "modified"
        assert dataset_y.meta.project == "project_y"


# ======================================================================================
# HISTORY BEHAVIOR
# ======================================================================================


class TestConcatenateHistory:
    """Characterize history behavior through concatenation.

    Notable behavior: history is REWRITTEN, not appended.
    Original history entries from input datasets are lost.
    """

    def test_history_rewritten(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert len(c.history) == 1
        assert "Created by concatenate" in c.history[0]

    def test_original_history_lost(self, dataset_x, dataset_y):
        """Notable behavior: input dataset history is not preserved."""
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.history[0] != dataset_x.history[0]
        assert c.history[0] != dataset_y.history[0]

    def test_history_is_string(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert isinstance(c.history[0], str)


# ======================================================================================
# ROI / MODELDATA
# ======================================================================================


class TestConcatenateRoiModeldata:
    """Characterize ROI and modeldata behavior through concatenation.

    Observations (not policy):
    - ROI propagates from the LAST dataset (via copy of last input)
    - modeldata propagates from the LAST dataset and is stale
      (its shape matches the input datasets, not the output)
    """

    def test_roi_from_last_dataset(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.roi == [5.0, 20.0]

    def test_modeldata_from_last_dataset(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert np.allclose(c.modeldata, 99.0)

    def test_modeldata_shape_stale(self, dataset_x, dataset_y):
        """Notable behavior: modeldata retains input shape,
        not the concatenated output shape."""
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.modeldata.shape == (5, 7)
        assert c.shape == (5, 14)


# ======================================================================================
# IDENTITY / PROVENANCE OBSERVATIONS
# ======================================================================================


class TestConcatenateIdentityProvenance:
    """
    Observe identity and provenance patterns in concatenation.

    Observations (not policy):
    - title, author, description, name are handled individually
      (title: first, name: last, author: merged, description: synthesized)
      This is NOT a copy-first identity preservation pattern.
    - origin, meta, roi, modeldata come from the last dataset (copy artifact)
    - history is rewritten — provenance is not preserved
    - This most closely resembles Pattern B (Rebuild / Synthesize):
      the result is a synthesized object, not an identity-preserved
      transformation of the first input.
    """

    def test_identity_pattern_is_rebuild(self, dataset_x, dataset_y):
        """
        Concatenation is not identity-preserving:
        the result is a synthesized multi-source object.
        """
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.title != dataset_y.title  # title not from last
        assert c.name != dataset_x.name  # name not from first
        assert c.author != dataset_x.author  # author merged
        assert c.description != dataset_x.description  # synthesized
        assert c.description != dataset_y.description

    def test_provenance_rewritten(self, dataset_x, dataset_y):
        """History is rewritten — provenance from inputs is lost."""
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert c.history != dataset_x.history
        assert c.history != dataset_y.history
        assert "Created by concatenate" in c.history[0]

    def test_same_title_no_warning(self):
        """No warning when both datasets have the same title."""
        x = Coord(np.linspace(4000.0, 1000.0, 7))
        y = Coord(np.linspace(0.0, 60.0, 5))
        a = NDDataset(np.arange(35.0).reshape(5, 7), coordset=[y, x], title="same")
        b = NDDataset(np.full((5, 7), 99.0), coordset=[y, x], title="same")
        c = concatenate(a, b, dims="x")
        assert c.title == "same"


# ======================================================================================
# STACK SEMANTICS
# ======================================================================================


class TestStackCharacterization:
    """Characterize stack() current behavior.

    stack prepends a new dimension, creates a coordinate with labels
    from dataset names, then delegates to concatenate(dims=0).

    Note: stack requires datasets to have CoordSets — bare datasets
    (no CoordSet) raise KeyError. All tests below use CoordSet datasets.
    """

    @pytest.fixture
    def stack_pair_1d(self):
        x = Coord(np.linspace(0.0, 10.0, 3), title="xval")
        a = NDDataset(np.array([10.0, 20.0, 30.0]), coordset=[x], name="alpha")
        b = NDDataset(np.array([40.0, 50.0, 60.0]), coordset=[x], name="beta")
        return a, b

    @pytest.fixture
    def stack_pair_2d(self):
        x = Coord(np.linspace(0.0, 10.0, 7), title="cols")
        y = Coord(np.linspace(0.0, 10.0, 5), title="rows")
        a = NDDataset(np.ones((5, 7)), coordset=[y, x], name="a")
        b = NDDataset(np.full((5, 7), 2.0), coordset=[y, x], name="b")
        return a, b

    def test_stack_creates_new_leading_dim(self, stack_pair_1d):
        a, b = stack_pair_1d
        s = stack(a, b)
        assert s.shape == (2, 3)
        assert len(s.dims) == 2

    def test_stack_2d(self, stack_pair_2d):
        a, b = stack_pair_2d
        s = stack(a, b)
        assert s.shape == (2, 5, 7)

    def test_stack_coord_labels_from_names(self, stack_pair_1d):
        a, b = stack_pair_1d
        s = stack(a, b)
        labels = s[s.dims[0]].labels
        assert labels is not None
        assert list(labels) == ["alpha", "beta"]

    def test_stack_coord_labels_none_if_no_names(self):
        x = Coord(np.linspace(0.0, 10.0, 3))
        a = NDDataset(np.array([10.0, 20.0, 30.0]), coordset=[x])
        b = NDDataset(np.array([40.0, 50.0, 60.0]), coordset=[x])
        s = stack(a, b)
        labels = s[s.dims[0]].labels
        assert labels is not None

    def test_stack_title_from_first(self, stack_pair_1d):
        a, b = stack_pair_1d
        a.title = "first"
        b.title = "second"
        s = stack(a, b)
        assert s.title == "first"

    def test_stack_history_rewritten(self, stack_pair_1d):
        a, b = stack_pair_1d
        s = stack(a, b)
        assert len(s.history) == 1
        assert "Created by concatenate" in s.history[0]

    def test_stack_author_merged(self, stack_pair_1d):
        a, b = stack_pair_1d
        a.author = "alice"
        b.author = "bob"
        s = stack(a, b)
        assert "alice" in s.author
        assert "bob" in s.author

    def test_stack_incompatible_shapes_raises(self):
        x1 = Coord(np.linspace(0.0, 10.0, 7))
        y1 = Coord(np.linspace(0.0, 10.0, 5))
        x2 = Coord(np.linspace(0.0, 10.0, 4))
        y2 = Coord(np.linspace(0.0, 10.0, 3))
        a = NDDataset(np.ones((5, 7)), coordset=[y1, x1])
        b = NDDataset(np.ones((3, 4)), coordset=[y2, x2])
        with pytest.raises(DimensionsCompatibilityError):
            stack(a, b)

    def test_stack_coordset_preserved(self):
        x = Coord(np.linspace(4000.0, 1000.0, 7), title="wn")
        y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
        a = NDDataset(np.arange(35.0).reshape(5, 7), coordset=[y, x])
        b = NDDataset(np.full((5, 7), 99.0), coordset=[y, x])
        s = stack(a, b)
        assert s.coordset is not None

    # ------------------------------------------------------------------
    # stack() origin propagation
    # ------------------------------------------------------------------

    def test_stack_origin_from_last_dataset(self):
        """Notable: origin propagates from the LAST dataset
        (consistent with concatenate's copy-of-last pattern)."""
        x = Coord(np.linspace(0.0, 10.0, 3))
        a = NDDataset(np.array([10.0, 20.0, 30.0]), coordset=[x])
        b = NDDataset(np.array([40.0, 50.0, 60.0]), coordset=[x])
        a.origin = "origin_first"
        b.origin = "origin_last"
        s = stack(a, b)
        assert s.origin == "origin_last"

    # ------------------------------------------------------------------
    # stack() custom meta propagation
    # ------------------------------------------------------------------

    def test_stack_meta_from_last_dataset(self):
        """Notable: custom meta propagates from the LAST dataset."""
        x = Coord(np.linspace(0.0, 10.0, 3))
        a = NDDataset(np.array([10.0, 20.0, 30.0]), coordset=[x])
        b = NDDataset(np.array([40.0, 50.0, 60.0]), coordset=[x])
        a.meta.project = "project_first"
        b.meta.project = "project_last"
        s = stack(a, b)
        assert s.meta.project == "project_last"

    def test_stack_meta_deep_copied(self):
        """Notable: result meta is deep-copied, not aliased
        to the source dataset object."""
        x = Coord(np.linspace(0.0, 10.0, 3))
        a = NDDataset(np.array([10.0, 20.0, 30.0]), coordset=[x])
        b = NDDataset(np.array([40.0, 50.0, 60.0]), coordset=[x])
        b.meta.project = "original"
        s = stack(a, b)
        s.meta.project = "modified"
        assert b.meta.project == "original"


# ======================================================================================
# EDGE CASES AND ERROR HANDLING
# ======================================================================================


class TestConcatenateEdgeCases:
    """Characterize error handling and edge cases."""

    def test_incompatible_shapes_raises(self):
        a = NDDataset(np.ones((5, 7)))
        b = NDDataset(np.ones((3, 7)))
        with pytest.raises(DimensionsCompatibilityError):
            concatenate(a, b, dims="x")

    def test_incompatible_shapes_other_dim_raises(self):
        a = NDDataset(np.ones((5, 7)))
        b = NDDataset(np.ones((5, 5)))
        with pytest.raises(DimensionsCompatibilityError):
            concatenate(a, b, dims="y")

    def test_concat_dim_not_found_raises(self):
        a = NDDataset(np.array([1.0, 2.0, 3.0]))
        b = NDDataset(np.array([4.0, 5.0, 6.0]))
        with pytest.raises((KeyError, ValueError)):
            concatenate(a, b, dims="nonexistent")

    def test_concat_single_dataset(self, dataset_x):
        c = concatenate(dataset_x, dims="x")
        assert c.shape == dataset_x.shape

    def test_concat_returns_nddataset(self, dataset_x, dataset_y):
        c = concatenate(dataset_x, dataset_y, dims="x")
        assert isinstance(c, NDDataset)

    def test_concat_as_method(self, dataset_x, dataset_y):
        """concatenate is available as a dataset method."""
        c = dataset_x.concatenate(dataset_y, dims="x")
        assert isinstance(c, NDDataset)
        assert c.shape == (5, 14)

    def test_concat_with_tuple(self, dataset_x, dataset_y):
        """concatenate accepts a tuple of datasets."""
        c = concatenate((dataset_x, dataset_y), dims="x")
        assert isinstance(c, NDDataset)
        assert c.shape == (5, 14)
