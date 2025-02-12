#!/usr/bin/env python3
"""
Generate configuration files for package dependencies and build settings.

This script automates the creation and maintenance of various configuration files
using pyproject.toml:
1. Conda environment files (environment.yml, environment_test.yml, environment_dev.yml)
2. Pip requirements files (requirements.txt, requirements_test.txt, requirements_dev.txt)


The script uses template files from the .ci/templates directory to generate the output files.
Dependencies are managed in a centralized way through these templates, ensuring consistency
across different package management systems (conda, pip) and development environments.

Notes
-----
    - All generated files include warnings about their auto-generated nature
    - Dependencies are automatically converted between conda and pip formats
    - Version specifications are properly handled across different formats
    - The script is designed to be run from the root of the repository and as a pre-commit hook.

"""

import re
from pathlib import Path

import toml
from jinja2 import Template

# Global variables
repo_path = Path(__file__).parent.parent
template_dir = repo_path / ".ci" / "templates"


def underline(s, indent=None):
    """
    Generate an underlined string for headers.

    Parameters
    ----------
    s : str
        The string to underline.
    indent : int, optional
        Number of spaces to indent the string.

    Returns
    -------
    str
        The underlined string.

    """
    x = "" if indent is None else " " * indent
    return f"{x}# {s}\n{x}# {'-' * len(s)}\n"


def pip2conda(package):
    """
    Convert a pip package specification to its conda equivalent.

    Parameters
    ----------
    package : str
        Pip package specification (e.g., "pandas==1.0")

    Returns
    -------
    str or None
        Conda package specification (e.g., "pandas=1.0") or None if package should be excluded

    Notes
    -----
    Handles two cases:
    1. Packages to rename (defined in RENAME)
    2. Version specifications (converting == to =)

    """
    # Mapping between conda and pip package names
    renaming = {  # pip to conda
        "gitpython": "git",  # Full package name
        "numpy-quaternion": "quaternion",  # Full package name
        "matplotlib": "matplotlib-base",  # Base package name
        "pypandoc_binary": "pypandoc",  # Full package name
    }
    # Convert pip version specifier to conda format
    package = re.sub("==", "=", package).strip()

    # Handle python spec
    package = package.split(";", maxsplit=1)
    package = (
        package[0]
        if len(package) < 2
        else (
            package[0]
            + " # "
            + package[1]
            .strip()
            .replace("python_version", "python")
            .replace("'", "")
            .replace('"', "")
            .replace(" ", "")
        )
    )

    # Handle version comparisons
    for compare in ("<=", ">=", "=", "<", ">"):
        if compare not in package:
            continue
        pkg, version = package.split(compare, maxsplit=1)
        pkg = pkg.strip()
        version = version.strip()
        if pkg in renaming:
            return "".join((renaming[pkg], compare, version))
        return "".join((pkg, compare, version))

    # compare was not found
    if package in renaming:
        return renaming[package]

    return package


def generate_conda_environments(deps, opt_deps):
    """
    Generate conda environment file(s).

    Parameters
    ----------
    deps : list
        List of core dependencies.
    opt_deps : dict
        Dictionary of optional dependencies.

    """
    # Define template paths
    req_template_file = template_dir / "environment.tmpl"
    template = Template(
        req_template_file.read_text("utf-8"),
        keep_trailing_newline=True,
    )

    # CORE dependencies
    deps_string = "\n" + underline("CORE dependencies", indent=4) + "    - "
    deps_string += "\n    - ".join([pip2conda(dep) for dep in deps])
    out = template.render(dependencies=deps_string)
    env_filename = repo_path / "environments" / "environment.yml"
    env_filename.write_text(out)

    # OPTIONAL dependencies
    for opt in opt_deps:
        opt_deps_string = (
            "\n" + underline(f"{opt.upper()} dependencies", indent=4) + "    - "
        )
        opt_deps_string += "\n    - ".join([pip2conda(dep) for dep in opt_deps[opt]])
        out = template.render(
            dependencies=deps_string
            if opt
            not in [
                "build",
            ]
            else "",
            optional_dependencies=opt_deps_string,
        )
        env_filename = repo_path / "environments" / f"environment_{opt}.yml"
        env_filename.write_text(out)


def generate_pip_requirements(deps, opt_deps):
    """
    Generate pip requirements files(s).

    Parameters
    ----------
    deps : list
        List of core dependencies.
    opt_deps : dict
        Dictionary of optional dependencies.

    """
    # Define template paths
    req_template_file = template_dir / "requirements.tmpl"
    template = Template(
        req_template_file.read_text("utf-8"),
        keep_trailing_newline=True,
    )

    # CORE dependencies
    deps_string = underline("CORE dependencies")
    deps_string += "\n".join(deps)
    out = template.render(dependencies=deps_string)
    req_filename = repo_path / "requirements" / "requirements.txt"
    req_filename.write_text(out)

    # OPTIONAL dependencies

    deps_string = underline("CORE dependencies")
    deps_string += "-r requirements.txt\n"

    for opt in opt_deps:
        if opt in [
            "build",
        ]:
            continue  # bypass this for pip
        opt_deps_string = underline(f"{opt.upper()} dependencies")
        opt_deps_string += "\n".join(opt_deps[opt])
        out = template.render(
            dependencies=deps_string,
            optional_dependencies=opt_deps_string,
        )
        req_filename = repo_path / "requirements" / f"requirements_{opt}.txt"
        req_filename.write_text(out)


def main():
    """Generate package configuration files."""
    # Load pyproject.toml
    pyproject_file = repo_path / "pyproject.toml"
    pyproject = toml.load(pyproject_file)

    # Get dependencies
    deps = pyproject["project"]["dependencies"]

    # Get optional dependencies
    opt_deps = pyproject["project"].get("optional-dependencies", {})

    # Generate requirements files for each category
    generate_pip_requirements(deps, opt_deps)

    # Generate environment files
    generate_conda_environments(deps, opt_deps)

    # Return success
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
