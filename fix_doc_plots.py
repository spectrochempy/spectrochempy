#!/usr/bin/env python
"""
AST-based script to transform standalone .plot() calls in documentation files.

Transforms:
    dataset.plot()
to:
    _ = dataset.plot()

Only transforms standalone expression statements (ast.Expr) where:
- The value is an ast.Call
- The call's func is an ast.Attribute
- The attribute name is 'plot'

Does NOT modify:
- Assignments (ax = dataset.plot())
- Return statements
- Call expressions used as part of other expressions
- Already prefixed lines (_ = dataset.plot())
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class PlotCallTransformer(ast.NodeTransformer):
    """
    AST transformer that converts standalone .plot() calls to _ = .plot()
    """

    def __init__(self):
        super().__init__()
        self.changes: List[Tuple[int, str, str]] = []

    def visit_Expr(self, node: ast.Expr):
        """
        Transform standalone plot() calls.
        """
        if isinstance(node.value, ast.Call):
            call = node.value

            if isinstance(call.func, ast.Attribute):
                attr = call.func

                if attr.attr == "plot":
                    lineno = call.lineno

                    assign = ast.Assign(
                        targets=[ast.Name(id="_", ctx=ast.Store())],
                        value=call,
                        lineno=lineno,
                        col_offset=0,
                    )
                    assign.lineno = lineno
                    assign.col_offset = 0

                    self.changes.append((lineno, "Expr", "Assign"))

                    return assign

        return node


def transform_file(
    filepath: Path, dry_run: bool = True
) -> Tuple[bool, List[Tuple[int, str, str]]]:
    """
    Transform a single Python file.
    """
    content = filepath.read_text(encoding="utf-8")

    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        print(f"  [SKIP] Syntax error in {filepath}: {e}")
        return False, []

    transformer = PlotCallTransformer()
    new_tree = transformer.visit(tree)

    if not transformer.changes:
        return False, []

    if dry_run:
        return True, transformer.changes

    # Apply changes - modify source text line by line
    lines = content.splitlines(keepends=True)
    new_lines = []
    change_set = set(c[0] for c in transformer.changes)

    for i, line in enumerate(lines, 1):
        if i in change_set:
            stripped = line.lstrip()
            if ".plot(" in stripped and not stripped.startswith("_ = "):
                # Preserve indentation, add _ = prefix
                indent = len(line) - len(stripped)
                new_line = " " * indent + "_ = " + stripped
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    new_content = "".join(new_lines)
    filepath.write_text(new_content, encoding="utf-8")

    return True, transformer.changes


def transform_rst_file(filepath: Path, dry_run: bool = True) -> Tuple[bool, int]:
    """
    Transform .rst files with >>> code blocks containing .plot() calls.
    """
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    changes = 0
    new_lines = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(">>> ") and ".plot(" in stripped:
            if "_ = " not in stripped:
                indent = len(line) - len(stripped)
                new_line = " " * indent + ">>> _ = " + stripped[4:]
                new_lines.append(new_line)
                changes += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if changes == 0:
        return False, 0

    if dry_run:
        return True, changes

    new_content = "".join(new_lines)
    filepath.write_text(new_content, encoding="utf-8")
    return True, changes


def get_files_to_process(docs_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Get all .py and .rst files in docs/sources directory."""
    py_files = []
    rst_files = []

    for pattern in ["**/*.py"]:
        for f in docs_dir.glob(pattern):
            if "examples/gallery" not in str(f):
                py_files.append(f)

    for pattern in ["**/*.rst"]:
        for f in docs_dir.glob(pattern):
            if "examples/gallery" not in str(f):
                rst_files.append(f)

    py_files.sort()
    rst_files.sort()
    return py_files, rst_files


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform standalone .plot() calls to _ = .plot() in docs"
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument(
        "--apply", dest="apply", action="store_true", help="Apply changes"
    )
    parser.add_argument("--docs-dir", type=str, default="docs/sources")
    parser.add_argument("--limit", type=int, default=5, help="Diffs to show")

    args = parser.parse_args()

    # If --apply is specified, disable dry-run
    dry_run = not args.apply

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"Error: Directory {docs_dir} does not exist")
        sys.exit(1)

    print(f"Scanning {docs_dir} for Python and RST files...")
    py_files, rst_files = get_files_to_process(docs_dir)
    print(f"Found {len(py_files)} Python files and {len(rst_files)} RST files\n")

    # Analyze Python file changes
    py_summary = {}
    for filepath in py_files:
        _, changes = transform_file(filepath, dry_run=True)
        if changes:
            py_summary[str(filepath)] = changes

    # Analyze RST file changes
    rst_summary = {}
    for filepath in rst_files:
        _, changes = transform_rst_file(filepath, dry_run=True)
        if changes:
            rst_summary[str(filepath)] = changes

    total_py = sum(len(c) for c in py_summary.values())
    total_rst = sum(rst_summary.values())
    total = total_py + total_rst

    if total == 0:
        print("No changes needed.")
        return

    print(f"Found {total} standalone .plot() calls to transform")
    print(f"  - Python: {total_py} in {len(py_summary)} files")
    print(f"  - RST: {total_rst} in {len(rst_summary)} files\n")

    if py_summary:
        print("Python files to modify:")
        for filepath, changes in py_summary.items():
            print(f"  {filepath}: {len(changes)} change(s)")
        print()

    if rst_summary:
        print("RST files to modify:")
        for filepath, changes in rst_summary.items():
            print(f"  {filepath}: {changes} change(s)")
        print()

    # Show sample
    print(f"Sample transformations (first {args.limit} files):")
    count = 0
    for filepath, changes in py_summary.items():
        if count >= args.limit:
            break
        print(f"\n{filepath}:")
        for lineno, old, new in changes[:3]:
            print(f"  Line {lineno}: {old} -> {new}")
        if len(changes) > 3:
            print(f"  ... and {len(changes) - 3} more")
        count += 1

    print("\n" + "=" * 60)

    if dry_run:
        print("DRY RUN - No changes were made.")
        print("Use --apply to actually modify the files.")
    else:
        print(f"Applying changes...")
        total_modified = 0

        for filepath_str, changes in py_summary.items():
            filepath = Path(filepath_str)
            modified, _ = transform_file(filepath, dry_run=False)
            if modified:
                total_modified += 1

        for filepath_str, changes in rst_summary.items():
            filepath = Path(filepath_str)
            modified, _ = transform_rst_file(filepath, dry_run=False)
            if modified:
                total_modified += 1

        print(f"Modified {total_modified} files.")


if __name__ == "__main__":
    main()
