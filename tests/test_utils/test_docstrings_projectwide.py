# SPDX-License-Identifier: BSD-3-Clause
# (see LICENSE.txt for details)

"""Project-wide guards against stale docrep-style source docstrings."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "spectrochempy"
DOCREP_DOCSTRING_MARKERS = ("%(", "_docstring.", "@_docstring")


def _iter_source_docstrings():
    for path in SOURCE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue

            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                continue

            yield path.relative_to(SOURCE_ROOT.parent), node, doc


def test_source_docstrings_contain_no_docrep_placeholders():
    offenders = []

    for path, node, doc in _iter_source_docstrings():
        if any(marker in doc for marker in DOCREP_DOCSTRING_MARKERS):
            name = getattr(node, "name", "<module>")
            offenders.append(f"{path}:{getattr(node, 'lineno', 1)}:{name}")

    assert offenders == []
