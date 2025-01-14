#!/usr/bin/env python3
"""
Create a local conda environment.yml file from a template
and convert it to the equivalent pip requirements.txt.

Additionally, it also creates the .conda/meta.yaml recipe.
Usage:

    Generate `requirements-dev.txt`
    $ python create_requirements.py [--dev] [--cantera]

Adapted from https://github.com/pandas-dev/pandas/scripts
/generate_pip_deps_from_conda.py (BSD 3-Clause License)

"""
import argparse
import re
from pathlib import Path

import yaml
from jinja2 import Template

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
    "renishaw_wire": "renishawWiRE",
}


def conda_package_to_pip(package):
    """
    Convert a conda package to its pip equivalent.

    In most cases they are the same, those are the exceptions:

    - Packages that should be excluded (in ``EXCLUDE``\ )
    - Packages that should be renamed (in ``RENAME``\ )
    - A package requiring a specific version, in conda is defined with a single
      equal (*e.g.* ``pandas=1.0``\ ) and in pip with two (*e.g.* ``pandas==1.0``\ )
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


def generate_pip_requirements(conda_fname, pip_fnames):
    """
    Generate the pip dependencies file from the conda file.

    Parameters
    ----------
    conda_fname : str
        Path to the conda file with dependencies (e.g. ``environment.yml`` ).
    pip_fnames : str or list of str
        Path to the pip file(s) with dependencies (e.g. ``requirements.txt`` ).

    Returns
    -------
    `bool`
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

    pip_content = header + "\n".join(pip_deps) + "\n"

    if not isinstance(pip_fnames, list):
        pip_fnames = [pip_fnames]

    for fname in pip_fnames:
        fname.write_text(pip_content)


if __name__ == "__main__":

    # setup
    parser = argparse.ArgumentParser(description="convert conda file to pip")

    parser.add_argument("--dash", help="use dash", action="store_true")
    parser.add_argument("--cantera", help="use cantera", action="store_true")
    parser.add_argument("--widgets", help="use widgets", action="store_true")
    args = parser.parse_args()

    repo_path = Path(__file__).parent.parent

    file_header = repo_path / ".ci" / "templates" / "environment.tmpl"
    file_dependencies = repo_path / ".ci" / "templates" / "dependencies.tmpl"
    file_meta = repo_path / ".ci" / "templates" / "meta.tmpl"

    template_header = Template(
        file_header.read_text("utf-8"), keep_trailing_newline=True
    )
    template_dependencies = Template(
        file_dependencies.read_text("utf-8"), keep_trailing_newline=True
    )

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

    # Generate environment.yml and requirements.txt files ..............................
    out_header = template_header.render(
        TEST=False,
        DEV=False,
        WIDGETS=args.widgets,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    out_dependencies = template_dependencies.render(
        TEST=False,
        DEV=False,
        WIDGETS=args.widgets,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    filename = repo_path / "environment.yml"
    filename.write_text(out_header + out_dependencies.rstrip() + "\n")

    generate_pip_requirements(filename, repo_path / "requirements" / "requirements.txt")

    # Generate requirements_test.txt files .............................................
    out_test_header = template_header.render(
        TEST=True,
        DEV=False,
        WIDGETS=args.widgets,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    out_test_dependencies = template_dependencies.render(
        TEST=True,
        DEV=False,
        WIDGETS=args.widgets,
        DASH=args.dash,
        CANTERA=args.cantera,
        COMMENT=comment,
    )
    filename = repo_path / "environment_test.yml"
    filename.write_text(out_test_header + out_test_dependencies.rstrip() + "\n")

    generate_pip_requirements(
        filename, repo_path / "requirements" / "requirements_test.txt"
    )

    # Generate environment_dev.yml and requirements_dev.txt files ......................
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

    generate_pip_requirements(
        filename, repo_path / "requirements" / "requirements_dev.txt"
    )

    # generate meta.yaml for conda recipe ..............................................
    out_meta = file_meta.read_text("utf-8")
    out_meta = out_meta.replace("DEPENDENCIES", out_dependencies.rstrip() + "\n")
    filename = repo_path / ".conda" / "meta.yaml"
    filename.write_text(out_meta)
