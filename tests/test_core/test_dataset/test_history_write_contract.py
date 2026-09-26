# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Static guard for direct mutation of detached NDDataset history views."""

from __future__ import annotations

import ast
from pathlib import Path


_MUTATING_METHODS = {
    "append",
    "extend",
    "insert",
    "clear",
    "pop",
    "remove",
    "sort",
    "reverse",
}
_VIEW_ATTRIBUTES = {"history", "history_entries"}
_ALLOWED_DIRECT_MUTATIONS = {
    (
        "src/spectrochempy/analysis/decomposition/mcrals.py",
        "self.history.append",
    ),
}


def _production_python_files(root):
    yield from (root / "src" / "spectrochempy").rglob("*.py")
    for plugin_source in (root / "plugins").glob("*/src"):
        yield from plugin_source.rglob("*.py")


def _direct_view_mutations(tree):
    """Yield manifest direct mutations; aliases are deliberately out of scope."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if (
                node.func.attr in _MUTATING_METHODS
                and isinstance(owner, ast.Attribute)
                and owner.attr in _VIEW_ATTRIBUTES
            ):
                yield node.lineno, ast.unparse(node.func)

        targets = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]

        for target in targets:
            if isinstance(node, ast.AugAssign) and isinstance(target, ast.Attribute):
                if target.attr in _VIEW_ATTRIBUTES:
                    yield node.lineno, ast.unparse(target)
            if isinstance(target, ast.Subscript):
                owner = target.value
                if isinstance(owner, ast.Attribute) and owner.attr in _VIEW_ATTRIBUTES:
                    yield node.lineno, ast.unparse(target)


def test_production_code_does_not_mutate_detached_history_views_directly():
    """Reject obvious ineffective writes without claiming full alias analysis.

    The single allowlisted call is the MCR-ALS model iteration history, not an
    ``NDDataset`` history. Public setters and legitimate canonical ``_history``
    storage, copy, and restoration remain outside this targeted guard.
    """
    root = Path(__file__).resolve().parents[3]
    unexpected = []

    for path in _production_python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(root).as_posix()
        for line, expression in _direct_view_mutations(tree):
            if (relative, expression) not in _ALLOWED_DIRECT_MUTATIONS:
                unexpected.append(f"{relative}:{line}: {expression}")

    assert unexpected == [], (
        "Direct mutation of a detached history view is ineffective. Use the public "
        "history writer methods instead:\n" + "\n".join(unexpected)
    )
