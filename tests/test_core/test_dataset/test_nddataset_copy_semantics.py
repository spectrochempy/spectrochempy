# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Focused characterization of NDDataset copy semantics."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.constants import MASKED
from spectrochempy.utils.testing import assert_array_equal
from spectrochempy.utils.testing import assert_units_equal


@pytest.fixture
def rich_dataset():
    """Small deterministic dataset carrying the mutable state copy must preserve."""
    y = Coord([0.0, 1.0], title="time", units="s")
    x = Coord(
        [4000.0, 3000.0, 2000.0],
        title="wavenumber",
        units="cm^-1",
        labels=["a", "b", "c"],
    )
    mask = np.zeros((2, 3), dtype=bool)
    mask[1, 2] = True

    dataset = NDDataset(
        np.arange(6.0, dtype="float64").reshape(2, 3),
        coordset=[y, x],
        units="absorbance",
        mask=mask,
        title="source spectrum",
        name="copy_source",
    )
    dataset.filename = Path("source.scp")
    dataset.author = "copy-audit"
    dataset.description = "copy semantics fixture"
    dataset.origin = "synthetic"
    dataset.meta.project = "copy-contract"
    dataset.meta.nested = {"steps": ["load", "copy"], "scale": {"factor": 1.0}}
    dataset.history = "initial processing"
    return dataset


COPY_SURFACES = [
    pytest.param(lambda dataset: dataset.copy(), id="copy_default"),
    pytest.param(lambda dataset: dataset.copy(deep=True), id="copy_deep_true"),
    pytest.param(lambda dataset: dataset.copy(deep=False), id="copy_deep_false"),
    pytest.param(copy.copy, id="copy_module"),
    pytest.param(copy.deepcopy, id="deepcopy_module"),
]


def _snapshot(dataset):
    return {
        "data": dataset.data.copy(),
        "mask": dataset.mask.copy(),
        "shape": dataset.shape,
        "dims": list(dataset.dims),
        "units": dataset.units,
        "y_data": dataset.y.data.copy(),
        "x_data": dataset.x.data.copy(),
        "y_units": dataset.y.units,
        "x_units": dataset.x.units,
        "x_labels": list(dataset.x.labels),
        "x_title": dataset.x.title,
        "meta_project": dataset.meta.project,
        "meta_nested": copy.deepcopy(dataset.meta.nested),
        "history": list(dataset.history),
        "name": dataset.name,
        "title": dataset.title,
        "filename": dataset.filename,
        "author": dataset.author,
        "description": dataset.description,
        "origin": dataset.origin,
    }


def _assert_matches_snapshot(dataset, snapshot):
    assert_array_equal(dataset.data, snapshot["data"])
    assert_array_equal(dataset.mask, snapshot["mask"])
    assert dataset.shape == snapshot["shape"]
    assert dataset.dims == snapshot["dims"]
    assert_units_equal(dataset.units, snapshot["units"])
    assert_array_equal(dataset.y.data, snapshot["y_data"])
    assert_array_equal(dataset.x.data, snapshot["x_data"])
    assert_units_equal(dataset.y.units, snapshot["y_units"])
    assert_units_equal(dataset.x.units, snapshot["x_units"])
    assert list(dataset.x.labels) == snapshot["x_labels"]
    assert dataset.x.title == snapshot["x_title"]
    assert dataset.meta.project == snapshot["meta_project"]
    assert dataset.meta.nested == snapshot["meta_nested"]
    assert dataset.history == snapshot["history"]
    assert dataset.name == snapshot["name"]
    assert dataset.title == snapshot["title"]
    assert dataset.filename == snapshot["filename"]
    assert dataset.author == snapshot["author"]
    assert dataset.description == snapshot["description"]
    assert dataset.origin == snapshot["origin"]


def _assert_public_copy_semantics(result, source):
    assert result is not source
    assert result == source
    assert result.shape == source.shape
    assert result.dims == source.dims
    assert_units_equal(result.units, source.units)
    assert_array_equal(result.data, source.data)
    assert_array_equal(result.mask, source.mask)
    assert_array_equal(result.y.data, source.y.data)
    assert_array_equal(result.x.data, source.x.data)
    assert_units_equal(result.y.units, source.y.units)
    assert_units_equal(result.x.units, source.x.units)
    assert list(result.x.labels) == list(source.x.labels)
    assert result.y.title == source.y.title
    assert result.x.title == source.x.title
    assert result.meta.project == source.meta.project
    assert result.meta.nested == source.meta.nested
    assert result.history == source.history
    assert result.name == source.name
    assert result.title == source.title
    assert result.filename == source.filename
    assert result.author == source.author
    assert result.description == source.description
    assert result.origin == source.origin


def _mutate_public_result_state(result):
    result[0, 0] = 99.0
    result[0, 1] = MASKED
    result.x.data[0] = 1234.0
    result.x.title = "changed coordinate"
    result.meta.project = "changed project"
    result.meta.nested["steps"].append("mutated")
    result.meta.nested["scale"]["factor"] = 2.0
    result.history = "mutation after copy"
    result.name = "changed_name"
    result.title = "changed title"
    result.filename = Path("changed.scp")
    result.author = "changed-author"
    result.description = "changed description"
    result.origin = "changed origin"


@pytest.mark.parametrize("copy_factory", COPY_SURFACES)
def test_copy_surfaces_preserve_public_dataset_semantics(
    rich_dataset,
    copy_factory,
):
    """All copy surfaces preserve documented identity and scientific content."""
    before = _snapshot(rich_dataset)

    result = copy_factory(rich_dataset)

    _assert_public_copy_semantics(result, rich_dataset)
    _assert_matches_snapshot(rich_dataset, before)


def test_default_copy_follows_deep_copy_observable_semantics(rich_dataset):
    """The documented default, deep=True, isolates public result mutations."""
    before = _snapshot(rich_dataset)

    result = rich_dataset.copy()
    _mutate_public_result_state(result)

    _assert_matches_snapshot(rich_dataset, before)


@pytest.mark.parametrize(
    "copy_factory",
    [
        pytest.param(lambda dataset: dataset.copy(deep=True), id="copy_deep_true"),
        pytest.param(copy.deepcopy, id="deepcopy_module"),
    ],
)
def test_deep_copy_surfaces_detach_mutable_scientific_state(
    rich_dataset,
    copy_factory,
):
    """Deep copy surfaces may assert targeted non-sharing for mutable payloads."""
    result = copy_factory(rich_dataset)

    assert not np.shares_memory(result.data, rich_dataset.data)
    assert not np.shares_memory(result.mask, rich_dataset.mask)
    assert result.coordset is not rich_dataset.coordset
    assert result.x is not rich_dataset.x
    assert not np.shares_memory(result.x.data, rich_dataset.x.data)
    assert result.meta is not rich_dataset.meta
    assert result.meta.nested is not rich_dataset.meta.nested
    assert result.meta.nested["steps"] is not rich_dataset.meta.nested["steps"]

    before = _snapshot(rich_dataset)
    _mutate_public_result_state(result)
    _assert_matches_snapshot(rich_dataset, before)


@pytest.mark.parametrize(
    "copy_factory",
    [
        pytest.param(lambda dataset: dataset.copy(deep=False), id="copy_deep_false"),
        pytest.param(copy.copy, id="copy_module"),
    ],
)
def test_shallow_copy_surfaces_preserve_source_under_public_result_mutations(
    rich_dataset,
    copy_factory,
):
    """Shallow copy must protect observable non-in-place mutation semantics."""
    before = _snapshot(rich_dataset)
    result = copy_factory(rich_dataset)

    _assert_public_copy_semantics(result, rich_dataset)
    _mutate_public_result_state(result)

    _assert_matches_snapshot(rich_dataset, before)


@pytest.mark.parametrize(
    "copy_factory",
    [
        pytest.param(lambda dataset: dataset.copy(), id="copy_default"),
        pytest.param(lambda dataset: dataset.copy(deep=True), id="copy_deep_true"),
        pytest.param(lambda dataset: dataset.copy(deep=False), id="copy_deep_false"),
        pytest.param(copy.copy, id="copy_module"),
        pytest.param(copy.deepcopy, id="deepcopy_module"),
    ],
)
def test_copy_surfaces_preserve_explicit_filename_none(copy_factory):
    """Explicit None filename state is preserved without normalizing on copy."""
    dataset = NDDataset([1.0, 2.0, 3.0], name="filename_none")
    dataset.filename = None
    before = _snapshot_minimal_filename(dataset)

    result = copy_factory(dataset)

    assert result is not dataset
    assert result.filename is None
    _assert_minimal_filename_snapshot(dataset, before)


def _snapshot_minimal_filename(dataset):
    return {
        "data": dataset.data.copy(),
        "name": dataset.name,
        "filename": dataset.filename,
    }


def _assert_minimal_filename_snapshot(dataset, snapshot):
    assert_array_equal(dataset.data, snapshot["data"])
    assert dataset.name == snapshot["name"]
    assert dataset.filename == snapshot["filename"]
