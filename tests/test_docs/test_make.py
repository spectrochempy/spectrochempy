# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the documentation build command."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_PATH = REPO_ROOT / "docs"
MAKE_PATH = DOCS_PATH / "make.py"


def _load_docs_make(monkeypatch):
    monkeypatch.syspath_prepend(str(DOCS_PATH))
    module_name = "_spectrochempy_docs_make_for_tests"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MAKE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_warning_is_error_cli_setting_is_preserved(monkeypatch):
    docs_make = _load_docs_make(monkeypatch)
    captured = {}

    class BuildDocumentationStub:
        def __init__(self, **kwargs):
            builder = object.__new__(docs_make.BuildDocumentation)
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
