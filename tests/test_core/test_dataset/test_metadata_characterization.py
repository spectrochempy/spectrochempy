# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Characterization tests for current NDDataset metadata propagation."""

from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
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
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    assert_units_preserved,
)
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    make_semantic_2d_dataset,
)


@pytest.fixture
def metadata_dataset():
    x_values = np.linspace(0.0, 4.0, 9)
    y_values = np.arange(4.0)
    xx, yy = np.meshgrid(x_values, y_values)
    return make_semantic_2d_dataset(
        data=xx**2 + 2.0 * yy,
        units="absorbance",
        title="test_title",
        name="sample_001",
        author="test_author",
        description="test_description",
        origin="test_origin",
        meta_project="metadata_contract",
        meta_instrument="test_instrument",
        meta_processing=["source"],
        filename=Path("test_filename.spc"),
        history="initial history marker",
        x_values=x_values,
        y_values=y_values,
    )


@pytest.fixture
def metadata_peak_dataset():
    dataset = scp.NDDataset(
        np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0]),
        coordset=[
            scp.Coord(np.linspace(0.0, 4.0, 9), title="wavelength", units="cm^-1")
        ],
        units="absorbance",
        title="test_title",
    )

    dataset.name = "sample_001"
    dataset.meta.project = "metadata_contract"
    dataset.meta.instrument = "test_instrument"
    dataset.author = "test_author"
    dataset.description = "test_description"
    dataset.origin = "test_origin"
    dataset.filename = Path("test_filename.spc")
    dataset.history = "initial history marker"

    return dataset


def test_copy_based_add_preserves_metadata(metadata_dataset):
    result = metadata_dataset + 1

    assert_basic_metadata_preserved(
        result,
        metadata_dataset,
        check_filename=True,
        meta_keys=("project", "instrument"),
    )
    assert_units_preserved(result, metadata_dataset)
    assert_dims_equal(result, ["y", "x"])
    assert_coordset_matches(result, metadata_dataset)
    assert_history_appended(result, metadata_dataset, "Binary operation add")


def test_copy_based_multiply_preserves_metadata(metadata_dataset):
    result = metadata_dataset * 2

    assert_basic_metadata_preserved(
        result,
        metadata_dataset,
        check_filename=True,
        meta_keys=("project", "instrument"),
    )
    assert_units_preserved(result, metadata_dataset)
    assert_dims_equal(result, ["y", "x"])
    assert_coordset_matches(result, metadata_dataset)
    assert_history_appended(result, metadata_dataset, "Binary operation mul")


def test_copy_based_abs_preserves_metadata(metadata_dataset):
    result = abs(metadata_dataset)

    assert_basic_metadata_preserved(
        result,
        metadata_dataset,
        check_filename=True,
        meta_keys=("project", "instrument"),
    )
    assert_units_preserved(result, metadata_dataset)
    assert_dims_equal(result, ["y", "x"])
    assert_coordset_matches(result, metadata_dataset)
    assert_history_appended(result, metadata_dataset, "Unary operation abs")


def test_mean_without_dimension_returns_quantity_without_dataset_metadata(
    metadata_dataset,
):
    result = metadata_dataset.mean()

    assert_units_preserved(result, metadata_dataset)
    assert not hasattr(result, "name")
    assert not hasattr(result, "title")
    assert not hasattr(result, "meta")
    assert not hasattr(result, "author")
    assert not hasattr(result, "description")
    assert not hasattr(result, "origin")
    assert not hasattr(result, "filename")
    assert not hasattr(result, "coordset")
    assert not hasattr(result, "history")


def test_mean_along_dimension_preserves_metadata_and_drops_reduced_coord(
    metadata_dataset,
):
    result = metadata_dataset.mean(dim="x")

    assert_basic_metadata_preserved(
        result,
        metadata_dataset,
        check_filename=True,
        meta_keys=("project", "instrument"),
    )
    assert_units_preserved(result, metadata_dataset)
    assert_dims_equal(result, ["y"])
    assert_coordset_matches(result, metadata_dataset, dims=("y",))
    assert "Initial history marker" in result.history[0]


def test_wrapper_based_filter_preserves_scientific_context_metadata(
    metadata_dataset,
):
    result = scp.Filter(method="savgol", size=5, order=2).transform(metadata_dataset)

    assert result.name == "sample_001_Filter.transform"
    assert result.title == metadata_dataset.title
    assert result.meta == metadata_dataset.meta
    assert result.meta is not metadata_dataset.meta
    assert result.meta.project == metadata_dataset.meta.project
    assert result.meta.instrument == metadata_dataset.meta.instrument
    assert result.meta.processing == metadata_dataset.meta.processing
    assert result.meta.processing is not metadata_dataset.meta.processing
    result.meta.processing.append("result")
    assert metadata_dataset.meta.processing == ["source"]
    assert_basic_metadata_preserved(
        result,
        metadata_dataset,
        check_name=False,
        check_title=False,
        check_filename=True,
        meta_keys=("project", "instrument"),
    )
    assert_units_preserved(result, metadata_dataset)
    assert_dims_equal(result, ["y", "x"])
    assert_coordset_matches(result, metadata_dataset)
    assert len(result.history) == 1
    assert "Created using method Filter.transform" in result.history[0]


def test_integrate_preserves_metadata_with_operation_overrides(metadata_dataset):
    result = metadata_dataset.trapezoid(dim="x")

    assert result.name == metadata_dataset.name
    assert result.title == "area"
    assert result.meta.project == metadata_dataset.meta.project
    assert result.meta.instrument == metadata_dataset.meta.instrument
    assert result.author == metadata_dataset.author
    assert result.description == "Integration of NDDataset 'sample_001' along dim: 'x'."
    assert result.origin == metadata_dataset.origin
    assert result.filename == metadata_dataset.filename
    assert result.units == metadata_dataset.units * metadata_dataset.x.units
    assert_dims_equal(result, ["y"])
    assert result.coordset is not None
    np.testing.assert_allclose(result.y.data, metadata_dataset.y.data)
    assert len(result.history) == 1
    assert (
        "Dataset resulting from application of `trapezoid` method" in result.history[0]
    )


def test_interpolate_preserves_metadata_and_replaces_interpolated_coord(
    metadata_dataset,
):
    new_x = np.linspace(0.5, 3.5, 4)

    result = metadata_dataset.interpolate(dim="x", coord=new_x)

    assert_basic_metadata_preserved(
        result,
        metadata_dataset,
        check_filename=True,
        meta_keys=("project", "instrument"),
    )
    assert_units_preserved(result, metadata_dataset)
    assert_dims_equal(result, ["y", "x"])
    assert_coordset_matches(result, metadata_dataset, dims=("y",))
    np.testing.assert_allclose(result.x.data, new_x)
    assert_history_appended(
        result,
        metadata_dataset,
        "Interpolated along dims ['x'] to 4 points using linear method",
    )


def test_dot_drops_meta_and_source_provenance(metadata_dataset):
    right = metadata_dataset.T
    right.name = "sample_002"
    right.title = "right_title"

    result = scp.dot(metadata_dataset, right)

    assert result.name != metadata_dataset.name
    assert result.name.startswith("NDDataset_")
    assert result.title == "test_title.right_title"
    assert result.meta.project is None
    assert result.meta.instrument is None
    assert result.author != metadata_dataset.author
    assert result.description == ""
    assert result.origin == ""
    assert result.filename != metadata_dataset.filename
    assert result.filename.suffix == ".scp"
    assert result.units == metadata_dataset.units * right.units
    assert_dims_equal(result, ["y", "x"])
    assert result.coordset is not None
    np.testing.assert_allclose(result.y.data, metadata_dataset.y.data)
    np.testing.assert_allclose(result.x.data, metadata_dataset.y.data)
    assert len(result.history) == 1
    assert "Dot product between sample_001 and sample_002" in result.history[0]


def test_find_peaks_preserves_metadata_with_name_and_history_overrides(
    metadata_peak_dataset,
):
    result, properties = scp.find_peaks(metadata_peak_dataset, height=0.5)

    assert properties is not None
    assert result.name == "peaks of sample_001"
    assert result.title == metadata_peak_dataset.title
    assert result.meta.project == metadata_peak_dataset.meta.project
    assert result.meta.instrument == metadata_peak_dataset.meta.instrument
    assert result.author == metadata_peak_dataset.author
    assert result.description == metadata_peak_dataset.description
    assert result.origin == metadata_peak_dataset.origin
    assert result.filename == metadata_peak_dataset.filename
    assert result.units == metadata_peak_dataset.units
    assert_dims_equal(result, ["x"])
    assert result.coordset is not None
    np.testing.assert_allclose(result.x.data, [0.5, 1.5, 2.5, 3.5])
    assert len(result.history) == 4  # initial + squeezed + slice + find_peaks
    assert "Initial history marker" in result.history[0]
    assert "Data squeezed" in result.history[1]
    assert "Slice extracted" in result.history[2]
    assert "Find_peaks(): 4 peak(s) found" in result.history[3]
