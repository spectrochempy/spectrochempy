# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Static checks for gallery example formatting."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
    "\n"
    "# %%\n"
    "\n"
    "# scp.show()\n"
)
DATADIR_TEST_VARIABLES = {"TEST_FILE", "TEST_FOLDER", "TEST_NMR_FOLDER"}


def _example_files() -> list[Path]:
    return sorted(
        path
        for root in EXAMPLE_ROOTS
        for path in root.rglob("*.py")
        if path.name.startswith("plot_") and ".ipynb_checkpoints" not in path.parts
    )


def _module_docstring_end(source: str) -> int | None:
    module = ast.parse(source)
    if (
        module.body
        and isinstance(module.body[0], ast.Expr)
        and isinstance(module.body[0].value, ast.Constant)
        and isinstance(module.body[0].value.value, str)
    ):
        return module.body[0].end_lineno

    return None


def _uses_datadir(node, names):
    return any(
        (
            isinstance(child, ast.Attribute)
            and child.attr == "datadir"
            and isinstance(child.value, ast.Attribute)
            and child.value.attr == "preferences"
            and isinstance(child.value.value, ast.Name)
            and child.value.value.id == "scp"
        )
        or (isinstance(child, ast.Name) and child.id in names)
        for child in ast.walk(node)
    )


def _is_scp_reader(func):
    if not isinstance(func, ast.Attribute) or not func.attr.startswith("read"):
        return False
    receiver = func.value
    while isinstance(receiver, ast.Attribute):
        receiver = receiver.value
    return isinstance(receiver, ast.Name) and receiver.id == "scp"


def _datadir_reader_calls(tree):
    assignments = [
        (node.value, node.targets if isinstance(node, ast.Assign) else [node.target])
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
    ]
    names = set()
    while True:
        derived = {
            target.id
            for value, targets in assignments
            if _uses_datadir(value, names)
            for assignment in targets
            for target in ast.walk(assignment)
            if isinstance(target, ast.Name)
        }
        if derived <= names:
            break
        names.update(derived)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _is_scp_reader(node.func)
        and any(
            _uses_datadir(arg, names)
            for arg in [*node.args, *(keyword.value for keyword in node.keywords)]
        )
    ]


def test_thumbnail_directives_follow_module_docstring():
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

        docstring_end = _module_docstring_end(source)
        first_cell = next(
            (index for index, line in enumerate(lines) if line.startswith("# %%")),
            None,
        )
        first_directive = directive_indexes[0]
        if (
            len(directive_indexes) != 1
            or docstring_end is None
            or first_directive < docstring_end
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
        if _datadir_reader_calls(tree):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

        for node in ast.walk(tree):
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
            if uses_os_path_join or uses_test_variable:
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert sorted(set(offenders)) == []


@pytest.mark.parametrize(
    ("source", "invalid"),
    [
        ('scp.read("irdata/input.spg")', False),
        ('scp.read(scp.preferences.datadir / "irdata/input.spg")', True),
        ('scp.read(filename=scp.preferences.datadir / "irdata/input.spg")', True),
        ('scp.NDDataset.read(scp.preferences.datadir / "irdata/input.spg")', True),
        (
            'filename = scp.preferences.datadir / "irdata/input.spg"\n'
            "scp.read_omnic(filename)",
            True,
        ),
        (
            "folder: Path = scp.preferences.datadir\n"
            'filename = folder / "irdata/input.spg"\n'
            "scp.read_omnic(filename)",
            True,
        ),
        (
            'filename = scp.preferences.datadir / "raw.txt"\n'
            "with filename.open('rb') as stream: table = np.loadtxt(stream)",
            False,
        ),
    ],
)
def test_relative_path_check_distinguishes_readers_from_raw_file_io(source, invalid):
    assert bool(_datadir_reader_calls(ast.parse(source))) is invalid


def test_thumbnail_configuration_is_recognized_by_sphinx_gallery():
    parser = pytest.importorskip("sphinx_gallery.py_source_parser")
    for path in _example_files():
        source = path.read_text(encoding="utf-8")
        directives = [
            line
            for line in source.splitlines()
            if line.strip().startswith(THUMBNAIL_PREFIXES)
        ]
        if not directives:
            continue
        configuration = parser.split_code_and_text_blocks(str(path))[0]
        for directive in directives:
            key, value = directive.removeprefix("# sphinx_gallery_").split("=", 1)
            assert configuration.get(key.strip()) == ast.literal_eval(
                value.strip()
            ), path
