"""Shared helpers for semantic baseline dataset construction and assertions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.constants import MASKED


def make_semantic_2d_dataset(
    *,
    data=None,
    title,
    name,
    units=None,
    author="test_author",
    description="test description",
    origin="test_origin",
    meta_project="test_project",
    meta_instrument=None,
    meta_processing=None,
    history="original entry",
    filename=None,
    x_labels=None,
    y_values=None,
    x_values=None,
):
    """Build a semantic-rich 2D dataset used across characterization suites."""
    y = Coord(
        np.linspace(0.0, 60.0, 5) if y_values is None else y_values,
        title="time",
        units="s",
    )
    x = Coord(
        np.linspace(4000.0, 1000.0, 7) if x_values is None else x_values,
        title="wavenumber",
        units="cm^-1",
        labels=x_labels,
    )
    if data is None:
        data = np.arange(35.0, dtype="float64").reshape(5, 7)

    dataset = NDDataset(data, coordset=[y, x], units=units, title=title, name=name)
    dataset.author = author
    dataset.description = description
    dataset.origin = origin
    dataset.meta.project = meta_project
    if meta_instrument is not None:
        dataset.meta.instrument = meta_instrument
    if meta_processing is not None:
        dataset.meta.processing = list(meta_processing)
    if filename is not None:
        dataset.filename = Path(filename)
    dataset.history = history
    return dataset


def mask_dataset_points(dataset, *indices):
    """Mask selected dataset elements using the current dataset masking policy."""
    for index in indices:
        dataset[index] = MASKED
    return dataset


def assert_basic_metadata_preserved(
    result,
    source,
    *,
    check_name=True,
    check_title=True,
    check_filename=False,
    meta_keys=("project",),
):
    """Assert the standard semantic metadata contract shared by many baselines."""
    if check_name:
        assert result.name == source.name
    if check_title:
        assert result.title == source.title
    assert result.author == source.author
    assert result.description == source.description
    assert result.origin == source.origin
    for key in meta_keys:
        assert getattr(result.meta, key) == getattr(source.meta, key)
    if check_filename:
        assert result.filename == source.filename


def assert_coordset_matches(
    result,
    source,
    *,
    dims=("y", "x"),
    check_titles=False,
    check_units=False,
):
    """Assert that selected coordinates were preserved from source to result."""
    assert result.coordset is not None
    for dim in dims:
        np.testing.assert_allclose(result[dim].data, source[dim].data)
        if check_titles:
            assert result[dim].title == source[dim].title
        if check_units:
            assert str(result[dim].units) == str(source[dim].units)
