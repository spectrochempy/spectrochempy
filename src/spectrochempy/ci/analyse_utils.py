# ruff: noqa: T201

import ast
from collections import defaultdict
from pathlib import Path


def analyze_methods():
    utils_path = Path("src/spectrochempy/utils")
    method_registry = defaultdict(list)

    print("# SpectroChemPy Utils Analysis\n")

    for py_file in sorted(utils_path.glob("*.py")):
        if py_file.stem.startswith("_"):
            continue

        print(f"\n## {py_file.stem}.py\n")

        try:
            with open(py_file) as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [arg.arg for arg in node.args.args]
                    docstring = ast.get_docstring(node)

                    print(f"### {node.name}")
                    print(f"- Line: {node.lineno}")
                    print(f"- Args: `{', '.join(args)}`")
                    if docstring:
                        print(f"- Doc: {docstring.split('\n')[0]}")

                    # Register method for duplicate checking
                    method_registry[node.name].append(
                        {"file": py_file.stem, "line": node.lineno, "args": args}
                    )

        except SyntaxError as e:
            print(f"Error parsing {py_file}: {e}")

    # Report potential duplicates
    print("\n## Potential Redundancies\n")
    for method, occurrences in method_registry.items():
        if len(occurrences) > 1:
            print(f"\n### `{method}`")
            for occurrence in occurrences:
                print(f"- {occurrence['file']}.py:{occurrence['line']}")
                print(f"  Args: `{', '.join(occurrence['args'])}`")


if __name__ == "__main__":
    analyze_methods()
