#!/usr/bin/env python3
"""
Generate configuration files for package dependencies and build settings.

This script automates the creation and maintenance of various configuration files:
1. Conda environment files (environment.yml, environment_test.yml, environment_dev.yml)
2. Pip requirements files (requirements.txt, requirements_test.txt, requirements_dev.txt)
3. Conda build recipe (meta.yaml)
4. Python package configuration (pyproject.toml)

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

import re
from pathlib import Path

import yaml
from jinja2 import Template

# Package management configuration
EXCLUDE = {
    "python",  # Handled by requires-python in pyproject.toml
    "pip",  # Package manager itself
    "spectrochempy_data",  # Internal package
    # "cantera",  # Optional dependency
    # "conda-build",  # Build-time only
    # "conda-verify",  # Build-time only
    # "anaconda-client",  # Build-time only
}

# Mapping between conda and pip package names
RENAME = {
    "git": "gitpython",  # Full package name
    "quaternion": "numpy-quaternion",  # Full package name
    "matplotlib-base": "matplotlib",  # Base package name
}

# Dependency configuration
DEPENDENCYCATEGORIES = ["core", "interactive", "test", "docs", "cantera", "dev"]


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

    # handle direct URL
    if "@git+" in package and package.startswith("pip"):
        return package.split("pip")[1].strip()[2:]

    # Handle package exclusions and renames
    if package in EXCLUDE:
        return

    if package in RENAME:
        return RENAME[package]

    return package


def generate_pip_requirements(
    case, conda_fname, pip_fnames, conda_deps, pip_deps, **kwargs
):
    """
    Generate pip requirements file(s) from conda environment file.

    Parameters
    ----------
    case : str
        The case for which to generate the requirements (e.g., "core", "test").
    conda_fname : Path
        Path to conda environment file (e.g., environment.yml)
    pip_fnames : Path or list of Path
        Path(s) to output pip requirements file(s)
    conda_deps : dict
        Dictionary to store conda dependencies
    pip_deps : dict
        Dictionary to store pip dependencies
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    tuple
        Updated conda_deps and pip_deps dictionaries

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
    case_dep = pip_deps.get(case, [])
    if case != "core":
        deps = conda_deps[case] = [dep for dep in deps if dep not in conda_deps["core"]]
    else:
        conda_deps[case] = deps

    for dep in deps:
        if isinstance(dep, str):
            pip_dep = conda_package_to_pip(dep)
            if pip_dep:
                case_dep.append(pip_dep)
        elif isinstance(dep, dict) and len(dep) == 1 and "pip" in dep:
            case_dep += dep["pip"]
        else:
            raise ValueError(f"Unexpected dependency {dep}")
    pip_deps[case] = case_dep

    # Generate header
    header = kwargs["COMMENT"]

    # Create output content
    pip_content = header + "\n"
    pip_content += (
        "\n# CORE dependencies\n"
        if case != "dev"
        else "\n# Development dependencies (include all)\n"
    )

    if case == "core":
        pip_content += "\n".join(pip_deps["core"]) + "\n"
    else:
        pip_content += "-r requirements.txt\n"
        for cas in DEPENDENCYCATEGORIES:
            if cas == "dev" and case == "dev":
                # include all other requirements (except cantera which is very specific and can be installed separately)
                for cas_ in DEPENDENCYCATEGORIES:
                    if cas_ not in ["dev", "cantera"]:
                        pip_content += f"-r requirements_{cas_}.txt\n"
            elif case == cas and cas != "dev":
                # write only the block for case
                pip_content += f"\n# {cas.upper()} dependencies\n"
                pip_content += "\n".join(pip_deps[cas]) + "\n"

    # Write to file(s)
    if not isinstance(pip_fnames, list):
        pip_fnames = [pip_fnames]

    for fname in pip_fnames:
        fname.write_text(pip_content)

    return conda_deps, pip_deps


def generate_pyproject_toml(filename, template_file, pip_deps):
    """
    Generate pyproject.toml file from template and dependencies.

    Parameters
    ----------
    filename : Path
        Path to output pyproject.toml file
    template_file : Path
        Path to template file for pyproject.toml
    pip_deps : dict
        Dictionary of pip dependencies

    Notes
    -----
    Generates pyproject.toml file with:
    1. Auto-generated header
    2. Dependencies for different sections (dev, test, core)
    """
    # Load template
    template = template_file.read_text("utf-8")

    def make_dep_string(case, no_brackets=False):
        strg = '[\n  "' if not no_brackets else '\n  "'
        strg += '",\n  "'.join(pip_deps.get(case, []))
        strg += '"\n]' if not no_brackets else '",\n'
        return strg

    # Define core dependencies
    core_deps = make_dep_string("core")

    # Define optional dependencies
    opt_deps = ""
    dev_deps = ""
    for case in DEPENDENCYCATEGORIES:
        if case not in ["core", "dev"]:
            opt_deps += f"{case}= {make_dep_string(case)}\n"

    # Add dev dependencies that include all others except cantera
    dev_deps = "dev = [\n"
    for case in DEPENDENCYCATEGORIES:
        if case not in ["core", "dev", "cantera"]:
            dev_deps += f"  # {case}{make_dep_string(case, no_brackets=True)}"
    dev_deps = dev_deps.rstrip(",\n") + "\n]"
    opt_deps += dev_deps

    # Replace placeholders in template
    pyproject_content = template.replace("DEPENDENCIES", core_deps).replace(
        "OPTIONAL", opt_deps
    )

    # Write pyproject.toml
    filename.write_text(pyproject_content)

    return


def generate_meta_yml(filename, template_file, conda_deps):
    """
    Generate meta.yaml file from template and dependencies.

    Parameters
    ----------
    filename : Path
        Path to output meta.yaml file
    template_file : Path
        Path to template file for meta.yaml
    conda_deps : dict
        Dictionary of conda dependencies

    Notes
    -----
    Generates meta.yaml file with:
    1. Auto-generated header
    2. Dependencies for different sections (core)
    """
    # Load template
    template = template_file.read_text("utf-8")

    def make_dep_string(case):
        strg = "    - "
        strg += "\n    - ".join(conda_deps.get(case, []))
        return strg

    # Define core dependencies
    core_deps = make_dep_string("core")

    # Replace placeholders in template
    meta_content = template.replace("DEPENDENCIES", core_deps)

    # Write meta.yaml
    filename.write_text(meta_content)

    return


def main():
    """
    Main function to generate package configuration files.
    """

    # Define template paths
    repo_path = Path(__file__).parent.parent
    template_dir = repo_path / ".ci" / "templates"
    header_template_file = template_dir / "environment.tmpl"
    dependencies_template_file = template_dir / "dependencies.tmpl"
    meta_template_file = template_dir / "meta.tmpl"
    pyproject_file_template = template_dir / "pyproject.tmpl"

    # Load template files
    template_header = Template(
        header_template_file.read_text("utf-8"), keep_trailing_newline=True
    )
    template_dependencies = Template(
        dependencies_template_file.read_text("utf-8"), keep_trailing_newline=True
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

    # Generate files for different cases
    pip_deps = {"core": []}  # initialise
    conda_deps = {"core": []}  # initialise

    for case in DEPENDENCYCATEGORIES:
        kwargs = {"COMMENT": comment}
        kwargs.update(
            {
                cat.upper(): False
                for cat in DEPENDENCYCATEGORIES
                if cat not in ["core", case]
            }
        )
        kwargs[case.upper()] = True
        kwargs["CORE"] = True

        out_header = template_header.render(**kwargs)
        out_dependencies = template_dependencies.render(**kwargs)

        # Generate environment file
        case_suffix = "" if case == "core" else f"_{case}"
        filename = repo_path / "environments" / f"environment{case_suffix}.yml"
        filename.write_text(out_header + out_dependencies.rstrip() + "\n")

        # Generate pip requirements file et set dict of dependencies
        req_filename = repo_path / "requirements" / f"requirements{case_suffix}.txt"
        conda_deps, pip_deps = generate_pip_requirements(
            case, filename, req_filename, conda_deps, pip_deps, **kwargs
        )

    # Generate pyproject.toml
    pyproject_filename = repo_path / "pyproject.toml"
    generate_pyproject_toml(pyproject_filename, pyproject_file_template, pip_deps)

    # Generate meta.yaml
    meta_filename = repo_path / "recipes" / "meta.yaml"
    generate_meta_yml(meta_filename, meta_template_file, conda_deps)

    # Print summary
    print("Generated files:")
    print(f"  - {pyproject_filename}")
    print(f"  - {meta_filename}")
    for case in DEPENDENCYCATEGORIES:
        case_suffix = "" if case == "core" else f"_{case}"
        print(f"  - {repo_path / 'requirements' / f'requirements{case_suffix}.txt'}")
        print(f"  - {repo_path / 'environments' / f'environment{case_suffix}.yml'}")

    # Return success
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
