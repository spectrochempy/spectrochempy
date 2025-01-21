# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Utility functions for printing version information.
"""

import locale
import platform
import re
import struct
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distributions, version
from os import environ
from pathlib import Path

import yaml

__all__ = ["show_versions"]


def show_versions(file=sys.stdout):
    """print the versions of spectrochempy and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    # from spectrochempy.application import version as scpversion

    underlined_title("SYSTEM INFO", "=", file=file)
    for key, val in _get_sys_info():
        print(f"- {key: <15} {val}", file=file)

    env = get_environment_info()
    underlined_title("ENVIRONMENT INFO", "=", file=file)
    for k, v in env.items():
        print(f"- {k: <15} {v}", file=file)

    underlined_title("SPECTROCHEMPY", "=", file=file)
    vers = version("spectrochempy")
    print(f"{'- version': <15} {vers}", file=file)

    # deps, deps_dev, deps_test = get_package_requirements()

    underlined_title("INSTALLED PACKAGES", "=", file=file)

    # dependencies
    installed = get_installed_versions()
    underlined_title("Dependencies", file=file)
    underlined_title(
        f"{'Package': <20} {'Required': <15} {'Installed': <15}", ".", ret=False
    )
    base, req = check_dependencies("", installed, env)
    print(base, file=file)
    underlined_title("Optional developpement dependencies", file=file)
    underlined_title(
        f"{'Package': <20} {'Required': <15} {'Installed': <15}", ".", ret=False
    )
    dev, _ = check_dependencies("dev", installed, exclude=req, env=env)
    print(dev, file=file)

    return


def underlined_title(s, char="-", file=sys.stdout, ret=True):
    n = "\n" if ret else ""
    print(n + s, file=file)
    print(char * len(s), file=file)


def _get_sys_info():
    """Returns system information as a dict"""
    # copied from XArray

    REPOS = Path(__file__).parent.parent.parent

    blob = []

    # get full commit hash
    commit = None
    if (REPOS / ".git").is_dir() and REPOS.is_dir():
        try:
            pipe = subprocess.Popen(
                'git log --format="%H" -n 1'.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            so, _ = pipe.communicate()
        except Exception:
            pass
        else:
            if pipe.returncode == 0:
                commit = so
                try:
                    commit = so.decode("utf-8")
                except ValueError:
                    pass
                commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    try:
        (sysname, _nodename, release, _version, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", sys.version),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", f"{sysname}"),
                ("OS-release", f"{release}"),
                ("machine", f"{machine}"),
                ("processor", f"{processor}"),
                ("byteorder", f"{sys.byteorder}"),
                ("LC_ALL", f"{environ.get('LC_ALL', 'None')}"),
                ("LANG", f"{environ.get('LANG', 'None')}"),
                ("LOCALE", f"{locale.getlocale()}"),
            ]
        )
    except Exception:
        pass

    return blob


def check_dependencies(run, installed, env, exclude=None):
    """Compare installed versions with requirements."""
    if env["type"] != "conda":
        req_file = f"requirements/requirements{'_' if run else ''}{run}.txt"
        requirements = parse_requirements(req_file)
    else:
        req_file = f"environment{'_' if run else ''}{run}.yml"
        requirements = parse_environment_yml(req_file)

    str = ""
    for pkg, req_ver in requirements.items():
        if exclude and pkg in exclude:
            continue
        try:
            inst_ver = version(pkg)
        except PackageNotFoundError:
            inst_ver = "Not installed"

        str += f"{pkg: <20} {req_ver or 'Any': <15} {inst_ver: <15}\n"

    return str, requirements


def get_user_directory():
    """Get user home directory path."""
    return str(Path.home())


def get_environment_info():
    """Detect virtual environment type and return info."""
    env_info = {}
    user_dir = get_user_directory()

    # Check for conda environment
    if environ.get("CONDA_DEFAULT_ENV") or environ.get("CONDA_PREFIX"):
        env_info["type"] = "conda"
        env_info["name"] = environ.get("CONDA_DEFAULT_ENV", "unknown")
        env_info["prefix"] = environ.get("CONDA_PREFIX", "unknown").replace(
            user_dir, "~"
        )

    # Check for pip virtual environment
    elif sys.prefix != sys.base_prefix:
        env_info["type"] = "venv"
        env_info["name"] = environ.get("VIRTUAL_ENV", "").split("/")[-1]
        env_info["prefix"] = str(Path(sys.prefix).relative_to(user_dir))

    else:
        env_info["type"] = "system"
        env_info["name"] = "none"
        env_info["prefix"] = str(Path(sys.prefix))

    return env_info


def parse_requirements(filename):
    """Parse requirements file and return dict of package names and version specs."""
    requirements = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Split package name and version spec
                parts = re.split(r"(>=|<=|==|<|>|~=)", line)
                if len(parts) > 1:
                    pkg = parts[0].strip()
                    ver_spec = "".join(parts[1:]).strip()
                    requirements[pkg] = ver_spec
                else:
                    requirements[line] = None
    return requirements


def parse_environment_yml(filename):
    """Parse environment.yml and return dict of package versions."""
    with open(filename) as f:
        env_dict = yaml.safe_load(f)

    requirements = {}
    for dep in env_dict.get("dependencies", []):
        if isinstance(dep, str) and not dep.startswith("pip:"):
            parts = re.split(r"(>=|<=|==|<|>|~=)", dep)
            pkg = parts[0].strip()
            ver_spec = "".join(parts[1:]).strip() if len(parts) > 1 else None
            requirements[pkg] = ver_spec
    return requirements


def get_installed_versions():
    """Get installed packages and their versions using importlib.metadata."""
    installed = {}
    for dist in distributions():
        try:
            installed[dist.metadata["Name"].lower()] = dist.version
        except Exception:
            # Skip packages with invalid metadata
            continue
    return installed


if __name__ == "__main__":  # pragma: no cover
    show_versions()
