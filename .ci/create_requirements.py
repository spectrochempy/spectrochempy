#!/usr/bin/env python3
"""
Create a local conda environment.yml file and convert to the equivalent pip requirements.txt.

Usage:

    Generate `requirements-dev.txt`
    $ python create_requirements.py --dev

Adapted from https://github.com/pandas-dev/pandas/scripts
/generate_pip_deps_from_conda.py (BSD 3-Clause License)

"""
import argparse
import re

import yaml

from jinja2 import Template
from pathlib import Path

EXCLUDE = {
    "python",
    "pip",
    "spectrochempy_data",
    "cantera",
    "conda-build",
    "conda-verify",
    "anaconda-client",
}
RENAME = {
    "pyqt": "pyqt5",
    "dask-core": "dask",
    "git": "gitpython",
    "quaternion": "numpy-quaternion",
    "matplotlib-base": "matplotlib",
    "nmrglue": "git+https://github.com/jjhelmus/nmrglue.git",
}


def conda_package_to_pip(package):
    """
    Convert a conda package to its pip equivalent.

    In most cases they are the same, those are the exceptions:
    - Packages that should be excluded (in `EXCLUDE`)
    - Packages that should be renamed (in `RENAME`)
    - A package requiring a specific version, in conda is defined with a single
      equal (e.g. ``pandas=1.0``) and in pip with two (e.g. ``pandas==1.0``)
    """
    package = re.sub("(?<=[^<>])=", "==", package).strip()

    for compare in ("<=", ">=", "=="):
        if compare not in package:
            continue

        pkg, version = package.split(compare)
        if pkg in EXCLUDE:
            return

        if pkg in RENAME:
            return "".join((RENAME[pkg], compare, version))

        break

    if package in EXCLUDE:
        return

    if package in RENAME:
        return RENAME[package]

    return package


def main(conda_fname, pip_fname):
    """
    Generate the pip dependencies file from the conda file.

    Parameters
    ----------
    conda_fname : str
        Path to the conda file with dependencies (e.g. `environment.yml`).
    pip_fname : str
        Path to the pip file with dependencies (e.g. `requirements.txt`).

    Returns
    -------
    bool
        True if the comparison fails, False otherwise.
    """
    with conda_fname.open() as conda_fd:
        deps = yaml.safe_load(conda_fd)["dependencies"]

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

    fname = conda_fname.name
    header = f"""# WARNING !!!
# ======================================================================================================================
# This file is auto-generated from {fname} file, do not modify it directly.
# See in `{fname}` for more information.
# ======================================================================================================================

"""

    pip_content = header + "\n".join(pip_deps) + "\n"

    pip_fname.write_text(pip_content)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="convert conda file to pip")

    parser.add_argument(
        "name",
        nargs="?",
        default="scpy",
        help="name of the environment (default scpy) ",
    )

    parser.add_argument("-v", "--version", help="Python version", default="3.8")
    parser.add_argument(
        "--dev", help="make a development environment", action="store_true"
    )
    parser.add_argument("--dash", help="use dash", action="store_true")
    parser.add_argument("--cantera", help="use cantera", action="store_true")

    args = parser.parse_args()

    repo_path = Path(__file__).parent.parent

    # generate environment yaml file
    tempfile = repo_path / ".ci" / "env_template.yml"
    template = Template(tempfile.read_text("utf-8"))

    name = args.name.split(".yml")[0]
    dev = "_dev" if args.dev else ""

    out = template.render(
        NAME=f"{name}{dev}",
        VERSION=args.version,
        DEV=args.dev,
        DASH=args.dash,
        CANTERA=args.cantera,
    )

    filename = repo_path / f"environment{dev}.yml"
    filename.write_text(out)

    # generate requirements
    main(
        filename,
        repo_path / f"requirements{dev}.txt",
    )
