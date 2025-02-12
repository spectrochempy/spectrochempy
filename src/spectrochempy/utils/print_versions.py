# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S602, S603

"""Utility functions for printing version information."""

import contextlib
import locale
import platform
import re
import struct
import subprocess
import sys
from importlib.metadata import distributions
from importlib.metadata import requires
from importlib.metadata import version
from os import environ
from pathlib import Path

__all__ = ["show_versions"]


def show_versions(file=sys.stdout):
    """
    Print the versions of spectrochempy and its dependencies.

    Parameters
    ----------
    file : file-like, optional
        Print to the given file-like object. Defaults to sys.stdout.

    """
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

    underlined_title("INSTALLED PACKAGES", "=", file=file)

    # dependencies

    # Load project metadata
    req = requires("spectrochempy")
    req = [r.split("; extra == ") for r in req]
    deps = [r[0] for r in req if len(r) == 1]
    opt_deps = {}
    for r in req:
        if len(r) == 1:
            continue
        if len(r) == 2:
            r[1] = r[1].strip('"')
            opt = opt_deps.get(r[1], [])
            opt.append(r[0])
            opt_deps[r[1]] = opt

    # Get installed packages
    installed = get_installed_versions()

    # display results of comparison with the requirements
    underlined_title("Dependencies", file=file)
    underlined_title(
        f"{'Package': <20} {'Required': <25} {'Installed': <15}",
        ".",
        ret=False,
        file=file,
    )
    strg = check_dependencies(deps, opt_deps, installed)
    print(strg, file=file)

    # import json

    # print(json.dumps(dict(sorted(installed.items())), indent=4))


def underlined_title(s, char="-", file=sys.stdout, ret=True):
    """
    Print an underlined title.

    Parameters
    ----------
    s : str
        The title string.
    char : str, optional
        The character to use for underlining. Defaults to '-'.
    file : file-like, optional
        Print to the given file-like object. Defaults to sys.stdout.
    ret : bool, optional
        Whether to add a newline before the title. Defaults to True.

    """
    n = "\n" if ret else ""
    print(n + s, file=file)
    print(char * len(s), file=file)


def _get_sys_info():
    """
    Return system information as a list of tuples.

    Returns
    -------
    list of tuples
        System information.

    """
    # copied from XArray

    REPOS = Path(__file__).parent.parent.parent

    blob = []

    # get full commit hash
    commit = None
    if (REPOS / ".git").is_dir() and REPOS.is_dir():
        try:
            git_executable = str(Path(sys.executable).parent / "git")
            if not Path(git_executable).is_file():
                raise FileNotFoundError(f"Git executable not found: {git_executable}")
            result = subprocess.run(  # noqa: S603
                [
                    git_executable,
                    "log",
                    "--format=%H",
                    "-n",
                    "1",
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            commit = result.stdout.strip()
        except Exception:  # noqa: S110
            pass
        else:
            if result.returncode == 0:
                commit = result.stdout
                with contextlib.suppress(ValueError):
                    commit = result.stdout
                commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    with contextlib.suppress(Exception):
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
            ],
        )

    return blob


def check_dependencies(deps, other_deps, installed):
    """
    Compare installed versions with requirements.

    Parameters
    ----------
    deps : list
        List of core dependencies.
    other_deps : dict
        Dictionary of optional dependencies.
    installed : dict
        Dictionary of installed packages and their versions.

    Returns
    -------
    str
        Formatted string of dependency comparison results.

    """
    # make a dictionary of package and version requirements
    requirements = {"core": deps, **other_deps}
    for key, deps in requirements.items():
        new_deps = {}
        for package in deps:
            # change eventual "==" to "="
            package = re.sub("==", "=", package).strip()
            # split version
            for compare in ("<=", ">=", "=", "@"):
                if compare not in package:
                    continue
                pkg, version = package.split(compare, maxsplit=1)
                if compare == "@":
                    version = version.strip()
                    version = version[0:4] + "..." + version[-17:]
                version = compare + version
                break
            else:
                pkg = package
                version = "Any"
            new_deps[pkg] = version
        requirements[key] = new_deps

    # compare with installed packages
    strg = ""
    for key, deps in requirements.items():
        strg += f"\n---- {key} ----\n"
        for pkg, req_ver in deps.items():
            inst_ver = installed.get(pkg, "Not installed")
            strg += f"{pkg: <20} {req_ver: <25} {inst_ver: <15}\n"

    return strg


def get_user_directory():
    """
    Get user home directory path.

    Returns
    -------
    str
        User home directory path.

    """
    return str(Path.home())


def get_environment_info():
    """
    Detect virtual environment type and return info.

    Returns
    -------
    dict
        Dictionary of environment information.

    """
    env_info = {}
    user_dir = get_user_directory()

    # Check for conda environment
    if environ.get("CONDA_DEFAULT_ENV") or environ.get("CONDA_PREFIX"):
        env_info["type"] = "conda"
        env_info["name"] = environ.get("CONDA_DEFAULT_ENV", "unknown")
        env_info["prefix"] = environ.get("CONDA_PREFIX", "unknown").replace(
            user_dir,
            "~",
        )

    # Check for pip virtual environment
    elif sys.prefix != sys.base_prefix:
        env_info["type"] = "venv"
        env_info["name"] = environ.get("VIRTUAL_ENV", "").split("/")[-1]
        env_info["prefix"] = str(Path(sys.prefix))  # .relative_to(user_dir))

    else:
        env_info["type"] = "system"
        env_info["name"] = "none"
        env_info["prefix"] = str(Path(sys.prefix))

    return env_info


def get_installed_versions():
    """
    Get installed packages and their versions using importlib.metadata.

    Returns
    -------
    dict
        Dictionary of installed packages and their versions.

    """
    installed = {}
    for dist in distributions():
        with contextlib.suppress(Exception):  # Skip packages with invalid metadata
            installed[dist.metadata["Name"].lower()] = dist.version
    return installed


if __name__ == "__main__":  # pragma: no cover
    show_versions()
