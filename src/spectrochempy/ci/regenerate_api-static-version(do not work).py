# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: T201
"""
Script to regenerate the SpectroChemPy API files during pre-commit.

This ensures API files are always up-to-date with the current version before commit.
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import jinja2

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

exclude = []  # "spectrochempy"]
exclude_startswith = ["~"]
exclude_within = [
    "spectrochempy.examples",
    "spectrochempy.extern",
    "spectrochempy.plugins",
    "spectrochempy.ci",
    "spectrochempy.data",
]


def find_python_files(directory, package_prefix="spectrochempy"):
    """Find all Python files in a directory and its subdirectories."""
    python_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, start=directory)

                # Convert file path to module path
                module_path = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                if file == "__init__.py":
                    # Handle __init__.py specially
                    module_path = os.path.dirname(rel_path).replace(os.sep, ".")

                if module_path:
                    full_module_path = (
                        f"{package_prefix}.{module_path}"
                        if module_path != "__init__"
                        else package_prefix
                    )

                    # Apply exclusion rules
                    if (
                        full_module_path in exclude
                        or any(full_module_path.startswith(f"{e}.") for e in exclude)
                        or any(
                            full_module_path.split(".")[-1].startswith(ss)
                            for ss in exclude_startswith
                        )
                        or any(ss in full_module_path for ss in exclude_within)
                    ):
                        print(f"Excluding: {full_module_path}")
                        continue

                    print("module_path:", full_module_path)
                    python_files.append((full_module_path, file_path))

    return sorted(python_files)


class ModuleAnalyzer(ast.NodeVisitor):
    """AST node visitor to analyze Python modules for API-related attributes."""

    def __init__(self):
        self.all_names = []
        self.dataset_methods = []
        self.configurables = []
        self.has_lazy_import = False

    def visit_Assign(self, node):
        """Visit assignment nodes to find API-related variables."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id == "__all__" and isinstance(node.value, ast.List):
                    # Extract items from __all__ list
                    for item in node.value.elts:
                        if isinstance(item, ast.Constant) and isinstance(
                            item.value, str
                        ):
                            self.all_names.append(item.value)
                elif target.id == "__dataset_methods__" and isinstance(
                    node.value, ast.List
                ):
                    # Extract dataset methods
                    for item in node.value.elts:
                        if isinstance(item, ast.Constant) and isinstance(
                            item.value, str
                        ):
                            self.dataset_methods.append(item.value)
                elif target.id == "__configurables__" and isinstance(
                    node.value, ast.List
                ):
                    # Extract configurables
                    for item in node.value.elts:
                        if isinstance(item, ast.Constant) and isinstance(
                            item.value, str
                        ):
                            self.configurables.append(item.value)

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Check for lazy import patterns in __getattr__ functions."""
        if node.name == "__getattr__" and any(
            isinstance(decorator, ast.Name) and decorator.id == "attach"
            for decorator in node.decorator_list
        ):
            self.has_lazy_import = True
        self.generic_visit(node)


def analyze_module_file(file_path):
    """Analyze a Python module file using AST parsing."""
    try:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)

        analyzer = ModuleAnalyzer()
        analyzer.visit(tree)
        return (
            analyzer.all_names,
            analyzer.dataset_methods,
            analyzer.configurables,
            analyzer.has_lazy_import,
        )
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], [], [], False


def create_api(version="", verbose=False):
    """
    Create the API files for SpectroChemPy using static analysis.

    Parameters
    ----------
    version : str, optional
        Version of SpectroChemPy for which the API is being generated.
        If not provided, no version information will be included.
    verbose : bool, optional
        Whether to print verbose output.

    Returns
    -------
    tuple
        A tuple containing the API methods, dataset methods, and configurable methods dictionaries.
    """
    # Setup Jinja2 environment
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True,
    )

    # Path where source files are located
    src_dir = project_root / "src" / "spectrochempy"

    # Output files
    scpy = project_root / "src" / "spectrochempy" / "lazyimport"
    apifile = scpy / "api_methods.py"
    datasetfile = scpy / "dataset_methods.py"
    # configurablefile = scpy / "configurable_classes.py"

    # Find and analyze Python files
    python_files = find_python_files(src_dir)

    _api_methods = {}
    _dataset_methods = {}
    _configurable_classes = {}

    for module_name, file_path in python_files:
        (
            all_names,
            dataset_methods,
            configurables,
            has_lazy_import,
        ) = analyze_module_file(file_path)

        # Skip __all__ from __init__.py if it's for lazy loading
        if Path(file_path).name == "__init__.py" and has_lazy_import:
            if verbose:
                print(f"Skipping lazy-loaded __all__ in {module_name}")
            continue

        # Process __all__ items
        for member in all_names:
            if verbose:
                print(module_name, "-> ", member)
            if member in _api_methods:
                print(module_name, f"Duplicate API method: {member} - skipping")
                continue
            _api_methods[member] = module_name

        # Process dataset methods
        for method in dataset_methods:
            if verbose:
                print(module_name, "-> ", method)
            _dataset_methods[method] = module_name

        # Process configurables
        for configurable in configurables:
            if configurable not in _configurable_classes:
                _configurable_classes[configurable] = module_name

    # Generate API methods file
    template = env.get_template("api_methods.py.tmpl")
    apifile.write_text(
        template.render(version=version, api_methods=_api_methods) + "\n"
    )

    # Generate dataset methods file
    template = env.get_template("dataset_methods.py.tmpl")
    datasetfile.write_text(
        template.render(version=version, dataset_methods=_dataset_methods) + "\n"
    )

    # # Generate configurable class file
    # template = env.get_template("configurable_classes.py.tmpl")
    # configurablefile.write_text(
    #     template.render(version=version, configurable_classes=_configurable_classes)
    #     + "\n"
    # )

    return _api_methods, _dataset_methods, _configurable_classes


def get_api_version():
    """
    Extract the version from the api_methods.py file.

    Returns
    -------
    str or None
        The version string if found, None otherwise.
    """
    try:
        root = project_root / "src" / "spectrochempy" / "lazyimport"
        apifile = root / "api_methods.py"

        if not apifile.exists():
            root.mkdir(parents=True, exist_ok=True)
            apifile.touch()
            return None

        content = apifile.read_text()

        # Look for the version line
        match = re.search(
            r"# This file was generated for SpectroChemPy version: (.*)", content
        )
        if match:
            return match.group(1)
        return None
    except Exception:
        return None


def should_regenerate_api(current_version):
    """
    Determine if the API files should be regenerated based on version comparison.

    Parameters
    ----------
    current_version : str
        The current version of SpectroChemPy.

    Returns
    -------
    bool
        True if regeneration is needed, False otherwise.
    """
    stored_version = get_api_version()

    # If we can't determine the stored version, regenerate to be safe
    if stored_version is None:
        return True

    # Compare versions
    return stored_version != current_version


class ArgumentParserWithErrorHelp(argparse.ArgumentParser):
    def error(self, message):
        """Print help message when an error occurs."""
        self.print_help()
        self.exit(2, f"\n{self.prog}: error: {message}\n")


def get_version_from_file():
    """Extract version from pyproject.toml using regex."""
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return "unknown"

    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return "unknown"


def main():
    """Regenerate the API files for SpectroChemPy."""
    # Parse command line arguments
    parser = ArgumentParserWithErrorHelp(
        description="Regenerate SpectroChemPy API files."
    )
    parser.add_argument(
        "--force",
        "-F",
        action="store_true",
        help="Force regeneration of API files regardless of version.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output."
    )

    args = parser.parse_args()

    try:
        # Extract version - using setuptools_scm dynamically
        from setuptools_scm import get_version

        version = get_version(root=str(project_root))

        # Check if regeneration is needed
        if should_regenerate_api(version) or args.force:
            print(f"Regenerating API files for version {version}")
            api_methods, dataset_methods, configurable_classes = create_api(
                version=version, verbose=args.verbose
            )
            print(
                f"API files regenerated successfully: "
                f"{len(api_methods)} API methods, "
                f"{len(dataset_methods)} dataset methods"
            )
        else:
            print("API files are already up to date")

        return 0

    except Exception as e:
        raise OSError(f"Error regenerating API files: {e}") from e


if __name__ == "__main__":
    sys.argv += ["-F"]
    sys.exit(main())
