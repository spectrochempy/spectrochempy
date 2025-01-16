#!/usr/bin/env python3
"""
Generate configuration files for package dependencies and build settings.

This script automates the creation and maintenance of various configuration files:
1. Conda environment files (environment.yml, environment_test.yml, environment_dev.yml)
2. Pip requirements files (requirements.txt, requirements_test.txt, requirements_dev.txt)
3. Conda build recipe (meta.yaml)
4. Python package configuration (pyproject.toml)

Usage:
    $ python create_requirements.py [--dev] [--cantera] [--dash]

Arguments:
    --dev     : Include development dependencies
    --cantera : Include Cantera-specific dependencies
    --dash    : Include Dash-specific dependencies

The script uses template files from the .ci/templates directory to generate the output files.
Dependencies are managed in a centralized way through these templates, ensuring consistency
across different package management systems (conda, pip) and development environments.

Notes:
    - All generated files include warnings about their auto-generated nature
    - Dependencies are automatically converted between conda and pip formats
    - Version specifications are properly handled across different formats

Adapted from: https://github.com/pandas-dev/pandas/scripts/generate_pip_deps_from_conda.py
License: BSD 3-Clause
"""

import argparse
import re
from pathlib import Path

import yaml
from jinja2 import Template

# Package management configuration
EXCLUDE = {
    "python",  # Handled by requires-python in pyproject.toml
    "pip",  # Package manager itself
    "spectrochempy_data",  # Internal package
    "cantera",  # Optional dependency
    "conda-build",  # Build-time only
    "conda-verify",  # Build-time only
    "anaconda-client",  # Build-time only
}

# Mapping between conda and pip package names
RENAME = {
    "pyqt": "pyqt5",  # Different naming convention
    "dask-core": "dask",  # Simplified pip name
    "git": "gitpython",  # Full package name
    "quaternion": "numpy-quaternion",  # Full package name
    "matplotlib-base": "matplotlib",  # Base package name
    "nmrglue": "git+https://github.com/jjhelmus/nmrglue.git",  # Direct from source
    "renishaw_wire": "renishawWiRE",  # Case difference
}


def conda_package_to_pip(package):
    """
    Convert a conda package specification to its pip equivalent.

    Parameters
    ----------
    package : str
        Conda package specification (e.g., "pandas=1.0")

    Returns
    -------
    str or None
        Pip package specification (e.g., "pandas==1.0") or None if package should be excluded

    Notes
    -----
    Handles three cases:
    1. Packages to exclude (defined in EXCLUDE)
    2. Packages to rename (defined in RENAME)
    3. Version specifications (converting = to ==)
    """
    # Convert conda version specifier to pip format
    package = re.sub("(?<=[^<>])=", "==", package).strip()

    # Handle version comparisons
    for compare in ("<=", ">=", "=="):
        if compare not in package:
            continue

        pkg, version = package.split(compare)
        if pkg in EXCLUDE:
            return

        if pkg in RENAME:
            return "".join((RENAME[pkg], compare, version))

        break

    # Handle package exclusions and renames
    if package in EXCLUDE:
        return

    if package in RENAME:
        return RENAME[package]

    return package


def generate_pip_requirements(conda_fname, pip_fnames):
    """
    Generate pip requirements file(s) from conda environment file.

    Parameters
    ----------
    conda_fname : Path
        Path to conda environment file (e.g., environment.yml)
    pip_fnames : Path or list of Path
        Path(s) to output pip requirements file(s)

    Returns
    -------
    list
        List of pip dependencies

    Notes
    -----
    Creates pip requirements file(s) with:
    1. Auto-generated header
    2. Converted conda dependencies
    3. Any pip-specific dependencies
    """
    # Load conda dependencies
    with conda_fname.open() as conda_fd:
        deps = yaml.safe_load(conda_fd)["dependencies"]

    # Convert dependencies to pip format
    pip_deps = []
    for dep in deps:
        if isinstance(dep, str):
            conda_dep = conda_package_to_pip(dep)
            if conda_dep:
                pip_deps.append(conda_dep)
        elif isinstance(dep, dict) and len(dep) == 1 and "pip" in dep:
            pip_deps += dep["pip"]
        else:
            raise ValueError(f"Unexpected dependency {dep}")

    # Generate header
    fname = conda_fname.name
    header = f"""
# ======================================================================================
#
# This file is auto-generated from {fname} file.
#
# !!! DO NOT MODIFY !!!
#
# See in `{fname}` for more information.
#
# ======================================================================================
""".lstrip()

    # Create output content
    pip_content = header + "\n".join(pip_deps) + "\n"

    # Write to file(s)
    if not isinstance(pip_fnames, list):
        pip_fnames = [pip_fnames]

    for fname in pip_fnames:
        fname.write_text(pip_content)

    return pip_deps


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Generate package configuration files from templates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dash", help="Include Dash dependencies", action="store_true")
    parser.add_argument(
        "--cantera", help="Include Cantera dependencies", action="store_true"
    )
    args = parser.parse_args()

    # Define paths
    repo_path = Path(__file__).parent.parent
    template_dir = repo_path / ".ci" / "templates"
    file_header = template_dir / "environment.tmpl"
    file_dependencies = template_dir / "dependencies.tmpl"
    file_meta = template_dir / "meta.tmpl"
    file_pyproject = template_dir / "pyproject.tmpl"

    # Load template files
    template_header = Template(
        file_header.read_text("utf-8"), keep_trailing_newline=True
    )
    template_dependencies = Template(
        file_dependencies.read_text("utf-8"), keep_trailing_newline=True
    )

    # Define standard warning header for generated files
    comment = """
# ======================================================================================
#
#       This file is automatically generated to be up-to-date in the master
#       repository.
#
#       !!! DO NOT MODIFY !!!
#
#       if you need to modify a dependency you need to follow these steps:
#
#       - Any change in dependencies must be first reflected in
#         file .ci/templates/dependencies.tmpl
#
#       - Then execute :
#          *  pre-commit run --all-files
#
# ======================================================================================
""".lstrip()

    # Generate standard environment files
    out_header = template_header.render(
        TEST=False,
        DEV=False,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    out_dependencies = template_dependencies.render(
        TEST=False,
        DEV=False,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    filename = repo_path / "environment.yml"
    filename.write_text(out_header + out_dependencies.rstrip() + "\n")

    deps = generate_pip_requirements(
        filename, repo_path / "requirements" / "requirements.txt"
    )

    # Generate test environment files
    out_test_header = template_header.render(
        TEST=True,
        DEV=False,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    out_test_dependencies = template_dependencies.render(
        TEST=True,
        DEV=False,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    filename = repo_path / "environment_test.yml"
    filename.write_text(out_test_header + out_test_dependencies.rstrip() + "\n")

    test_deps = generate_pip_requirements(
        filename, repo_path / "requirements" / "requirements_test.txt"
    )

    # Generate development environment files
    out_dev_header = template_header.render(
        TEST=True,
        DEV=True,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    out_dev_dependencies = template_dependencies.render(
        TEST=True,
        DEV=True,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    filename = repo_path / "environment_dev.yml"
    filename.write_text(out_dev_header + out_dev_dependencies.rstrip() + "\n")

    dev_deps = generate_pip_requirements(
        filename, repo_path / "requirements" / "requirements_dev.txt"
    )

    # Generate conda recipe meta.yaml
    out_meta = file_meta.read_text("utf-8")
    out_meta = out_meta.replace("DEPENDENCIES", out_dependencies.rstrip() + "\n")
    filename = repo_path / ".conda" / "meta.yaml"
    filename.write_text(out_meta)

    # Load templates
    template_pyproject = file_pyproject.read_text("utf-8")

    # Helper function to format dependency lists for TOML
    def format_deps(deps):
        """Format dependency list for TOML output with proper indentation."""
        return (
            str(deps)
            .replace("'", '"')
            .replace(",", ",\n\t")
            .replace("[", "[\n\t")
            .replace("]", "\n]")
        )

    # Process dependencies for different sections
    dev_deps = [dep for dep in dev_deps if dep not in deps]
    test_deps = [dep for dep in test_deps if dep not in deps]

    # Replace placeholders in template
    pyproject_content = (
        template_pyproject.replace("DEV_DEPENDENCIES", format_deps(dev_deps))
        .replace("TEST_DEPENDENCIES", format_deps(test_deps))
        .replace("DEPENDENCIES", format_deps(deps))
    )

    # Write pyproject.toml
    filename = repo_path / "pyproject.toml"
    filename.write_text(pyproject_content)
