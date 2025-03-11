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
import sys
from pathlib import Path
from pkgutil import walk_packages

import jinja2
from traitlets import import_item

import spectrochempy  # noqa: F401

# Add the project root to the path so we can import spectrochempy modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

exclude = ["spectrochempy"]
exclude_startswith = ["~"]
exclude_within = [
    "spectrochempy.examples",
    "spectrochempy.extern",
    "spectrochempy.plugins",
    "spectrochempy.ci",
    "spectrochempy.data",
]


def list_packages(package):
    """
    Return a list of the names of a package and its subpackages.

    This only works if the package has a :attr:`__path__` attribute, which is
    not the case for some (all?) of the built-in packages.
    """
    # Based on response at
    # http://stackoverflow.com/questions/1707709.

    names = []  # package.__name__]
    for __, name, __ in walk_packages(
        package.__path__, prefix=package.__name__ + ".", onerror=lambda x: None
    ):
        s = name.split(".")[-1]
        if (
            name in exclude
            or any(s.startswith(ss) for ss in exclude_startswith)
            or any(ss in name for ss in exclude_within)
        ):
            print(f"Excluding: {name}")
            continue
        print("name:", name)
        names.append(name)

    return sorted(names)


def create_api(version="", verbose=False):
    """
    Create the API files for SpectroChemPy.

    Parameters
    ----------
    version : str, optional
        Version of SpectroChemPy for which the API is being generated.
        If not provided, no version information will be included.

    Returns
    -------
    tuple
        A tuple containing the API methods, dataset methods, and configurable methods dictionaries.
    """
    # Setup Jinja2 environment
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(  # noqa: S701
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    scpy = project_root / "src" / "spectrochempy" / "lazyimport"
    apifile = scpy / "api_methods.py"
    datasetfile = scpy / "dataset_methods.py"
    # configurablefile = scpy / "configurable_classes.py"  # New file for configurables

    modules = list_packages(sys.modules["spectrochempy"])
    _api_methods = {}
    _dataset_methods = {}
    _configurable_classes = {}  # New dictionary for configurables

    for module in modules:
        try:  # with contextlib.suppress(ImportError):
            x = import_item(module)
        except ImportError:
            print(f"Could not import module: {module}")
            continue
        members = []
        if hasattr(x, "__all__") and not (
            Path(x.__file__).name == "__init__.py"
            and hasattr(x, "__getattr__")
            and x.__getattr__.__qualname__ == "attach.<locals>.__getattr__"
        ):
            # do not take __all__ from __init__.py files if they are for lazy loading
            members = x.__all__
        for member in members:
            if verbose:
                print(module, "-> ", member)
            if member in _api_methods:
                print(module, f"Duplicate API method: {member} - skipping")
                continue
            _api_methods[member] = module

        methods = []
        if hasattr(x, "__dataset_methods__"):
            methods = x.__dataset_methods__
        for method in methods:
            if verbose:
                print(module, "-> ", methods)
            _dataset_methods[method] = module

        # Extract configurable classes
        configurables = []
        if hasattr(x, "__configurables__"):
            configurables = x.__configurables__
        for configurable in configurables:
            if configurable in _configurable_classes:
                # print(module, f"Duplicate configurable: {configurable} - skipping")
                continue
            _configurable_classes[configurable] = module

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
        import re

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
        # Get the current version
        # version_file = project_root / "pyproject.toml"
        # if not version_file.exists():
        #     print("Could not find pyproject.toml")
        #     sys.exit(1)

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
                f"{len(dataset_methods)} dataset methods, "
                # f"{len(configurable_classes)} configurable classes"
            )
        else:
            print("API files are already up to date")

        return 0

    except Exception as e:
        raise OSError(f"Error regenerating API files: {e}") from e


if __name__ == "__main__":
    sys.argv += ["-F"]
    sys.exit(main())
