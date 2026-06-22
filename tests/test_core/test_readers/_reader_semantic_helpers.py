"""Lightweight helpers for reader semantic characterization baselines."""

from __future__ import annotations

from datetime import datetime


def assert_dataset_identity(
    dataset,
    *,
    name=None,
    title=None,
    units=None,
):
    """Assert the dataset-level scientific identity fields."""
    if name is not None:
        assert dataset.name == name
    if title is not None:
        assert dataset.title == title
    if units is not None:
        assert dataset.units == units


def assert_dataset_provenance(
    dataset,
    *,
    filename_name=None,
    origin=None,
    description_contains=None,
    acquisition_date_present=None,
):
    """Assert dataset-level provenance fields without over-specifying behavior."""
    if filename_name is not None:
        assert dataset.filename.name == filename_name
    if origin is not None:
        assert dataset.origin == origin
    if description_contains is not None:
        assert description_contains in dataset.description
    if acquisition_date_present is True:
        assert isinstance(dataset.acquisition_date, datetime)
    elif acquisition_date_present is False:
        assert dataset.acquisition_date is None


def assert_coordinate_semantics(
    dataset,
    dim,
    *,
    size=None,
    title=None,
    units=None,
):
    """Assert the main semantic properties of one coordinate and return it."""
    coord = dataset.coord(dim)
    assert coord is not None
    if size is not None:
        assert coord.size == size
    if title is not None:
        assert coord.title == title
    if units is not None:
        assert str(coord.units) == units
    return coord


def assert_label_structure(
    coord,
    *,
    shape=None,
):
    """Assert that labels are present and optionally match an expected shape."""
    labels = coord.labels
    assert labels is not None
    if shape is not None:
        assert labels.shape == shape
    return labels


def assert_history_present(dataset, *substrings):
    """Assert that history is present and contains the requested semantic markers."""
    assert dataset.history
    history_text = " ".join(str(entry) for entry in dataset.history)
    for substring in substrings:
        assert substring in history_text


def assert_meta_keys_present(dataset, *keys):
    """Assert selected metadata keys are retained in the dataset Meta payload."""
    for key in keys:
        assert key in dataset.meta
