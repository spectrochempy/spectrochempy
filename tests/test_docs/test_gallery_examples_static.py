# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Static checks for gallery example formatting."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOTS = (
    REPO_ROOT / "src" / "spectrochempy" / "examples",
    *sorted((REPO_ROOT / "plugins").glob("spectrochempy-*/examples")),
)
THUMBNAIL_PREFIXES = (
    "# sphinx_gallery_thumbnail_number",
    "# sphinx_gallery_thumbnail_path",
)
CANONICAL_FOOTER = (
    "# %%\n"
    "# Uncomment the following line to display all figures when running the script\n"
    "# directly with Python.\n"
    "#\n"
    "# # scp.show()\n"
)
DATADIR_TEST_VARIABLES = {"TEST_FILE", "TEST_FOLDER", "TEST_NMR_FOLDER"}


def _example_files() -> list[Path]:
    return sorted(
        path
        for root in EXAMPLE_ROOTS
        for path in root.rglob("*.py")
        if path.name.startswith("plot_") and ".ipynb_checkpoints" not in path.parts
    )


def _module_docstring_start(lines: list[str], source: str) -> int | None:
    module = ast.parse(source)
    if (
        module.body
        and isinstance(module.body[0], ast.Expr)
        and isinstance(module.body[0].value, ast.Constant)
        and isinstance(module.body[0].value.value, str)
    ):
        return module.body[0].lineno

    for lineno, line in enumerate(lines, start=1):
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            return lineno
    return None


def test_thumbnail_directives_precede_module_docstring():
    misplaced = []
    for path in _example_files():
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        directive_indexes = [
            index
            for index, line in enumerate(lines)
            if line.strip().startswith(THUMBNAIL_PREFIXES)
        ]
        if not directive_indexes:
            continue

        docstring_start = _module_docstring_start(lines, source)
        first_cell = next(
            (index for index, line in enumerate(lines) if line.startswith("# %%")),
            None,
        )
        first_directive = directive_indexes[0]
        ruff_index = next(
            (index for index, line in enumerate(lines) if line.startswith("# ruff:")),
            None,
        )
        if (
            len(directive_indexes) != 1
            or docstring_start is None
            or first_directive >= docstring_start - 1
            or (ruff_index is not None and first_directive != ruff_index + 1)
            or (first_cell is not None and first_directive > first_cell)
        ):
            misplaced.append(path.relative_to(REPO_ROOT).as_posix())

    assert misplaced == []


def test_gallery_examples_use_canonical_show_footer():
    offenders = []
    for path in _example_files():
        source = path.read_text(encoding="utf-8")
        if "scp.show()" not in source:
            continue
        if not source.endswith(CANONICAL_FOOTER):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
        assert "This ends the example" not in source
        assert "# scp.show()  #" not in source
        assert "\nscp.show()\n" not in source

    assert offenders == []


def test_gallery_examples_use_datadir_relative_paths():
    offenders = []
    for path in _example_files():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            uses_scp_datadir = (
                isinstance(node, ast.Attribute)
                and node.attr == "datadir"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "preferences"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "scp"
            )
            uses_os_path_join = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "path"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os"
            )
            uses_test_variable = (
                isinstance(node, ast.Name) and node.id in DATADIR_TEST_VARIABLES
            )
            if uses_scp_datadir or uses_os_path_join or uses_test_variable:
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert sorted(set(offenders)) == []
