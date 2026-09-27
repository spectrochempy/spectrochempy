# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the documentation build command."""

from __future__ import annotations

import importlib
import sys


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
