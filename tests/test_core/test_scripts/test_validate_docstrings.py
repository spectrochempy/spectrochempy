"""Tests for the project docstring validator."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from spectrochempy.ci import validate_docstrings as vd


def _good_docstring(value):
    """
    Return a value.

    Parameters
    ----------
    value : array-like
        Value to return.

    Returns
    -------
    array-like
        The supplied value.
    """
    return value


def _bad_docstring(value):
    """
    Return a value.

    Parameters
    ----------
    value : array_like
        Value to return.

    Returns
    -------
    array_like
        The supplied value.
    """
    return value


def test_reference_discovery_uses_maintained_sources(monkeypatch, tmp_path):
    repository_root = Path(__file__).resolve().parents[3]
    installed_module = (
        tmp_path / "site-packages" / "spectrochempy" / "ci" / "validate_docstrings.py"
    )
    monkeypatch.setattr(vd, "__file__", str(installed_module))
    monkeypatch.chdir(repository_root / "tests" / "test_core")

    files = vd.discover_api_reference_files()

    assert {path.name for path in files} >= {"index.rst", "plugins.rst"}
    assert all(path.parent.name == "reference" for path in files)
    assert all(path.parent.parent.name == "sources" for path in files)


def test_reference_discovery_rejects_missing_sources(tmp_path):
    with pytest.raises(RuntimeError, match="No public API reference sources"):
        vd.discover_api_reference_files(tmp_path)


def test_api_parser_resolves_relative_and_fully_qualified_names():
    source = io.StringIO(
        """.. currentmodule:: spectrochempy

.. autosummary::

    NDDataset

.. autosummary::

    spectrochempy.Coord
"""
    )

    items = list(vd.get_api_items(source))

    assert [item[0] for item in items] == [
        "spectrochempy.NDDataset",
        "spectrochempy.Coord",
    ]
    assert items[0][1] is vd.spectrochempy.NDDataset
    assert items[1][1] is vd.spectrochempy.Coord


def test_validate_all_rejects_empty_inventory(tmp_path):
    (tmp_path / "index.rst").write_text("Public API\n==========\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="empty validation inventory"):
        vd.validate_all(None, api_reference_path=tmp_path)


def test_known_good_and_bad_docstrings(monkeypatch):
    objects = {"good": _good_docstring, "bad": _bad_docstring}
    monkeypatch.setattr(
        vd.Validator,
        "_load_obj",
        staticmethod(objects.__getitem__),
    )

    good_errors = vd.spectrochempy_validate("good")["errors"]
    bad_errors = vd.spectrochempy_validate("bad")["errors"]

    assert not any(code == "GL05" for code, _ in good_errors)
    assert any(code == "GL05" for code, _ in bad_errors)


def test_examples_are_linted_with_ruff():
    raw_doc = "\n".join(
        [
            "Demonstrate example linting.",
            "",
            "Examples",
            "--------",
            ">>> missing_name",
        ]
    )

    def example():
        pass

    example.__doc__ = raw_doc
    doc = vd.spectrochempyDocstring(
        "example",
        vd.get_doc_object(example),
    )

    errors = list(doc.validate_pep8())

    assert any(code == "F821" for code, _, _ in errors)
