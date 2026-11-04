# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the documentation build command."""

from __future__ import annotations

import importlib
import sys

import pytest


def _load_docs_make():
    """Import ``docs.make`` from the repository root, the way the workflows do."""
    return importlib.import_module("docs.make")


def test_warning_is_error_cli_setting_is_preserved(monkeypatch):
    docs_make = _load_docs_make()
    build_documentation = docs_make.BuildDocumentation
    captured = {}

    class BuildDocumentationStub:
        def __init__(self, **kwargs):
            builder = object.__new__(build_documentation)
            captured.update(builder._init_settings(kwargs))

        def html(self):
            return 0

    monkeypatch.setattr(docs_make, "BuildDocumentation", BuildDocumentationStub)
    monkeypatch.setattr(
        sys, "argv", ["make.py", "--warning-is-error", "-j", "1", "html"]
    )

    result = docs_make._main()

    assert result == 0
    assert captured["warningiserror"] is True


@pytest.mark.parametrize("notebook_exists", [False, True])
def test_del_nb_synchronizes_missing_and_existing_notebooks(
    tmp_path, monkeypatch, notebook_exists
):
    docs_make = _load_docs_make()
    builder = object.__new__(docs_make.BuildDocumentation)
    builder.SRC = tmp_path
    builder.settings = builder._init_settings({"delnb": True})
    builder.singledoc = None
    builder.directory = None
    script = tmp_path / "tutorial.py"
    script.write_text("# %%\nprint('tutorial')\n", encoding="utf-8")
    notebook = script.with_suffix(".ipynb")
    if notebook_exists:
        notebook.write_text("old notebook", encoding="utf-8")
    synchronized = []

    def sync_pair(item):
        assert not notebook.exists()
        synchronized.append(item)
        notebook.write_text("regenerated notebook", encoding="utf-8")

    monkeypatch.setattr(builder, "_sync_notebook_pair", sync_pair)

    builder._sync_notebooks()

    assert synchronized == [script.with_suffix("")]
    assert notebook.read_text(encoding="utf-8") == "regenerated notebook"
