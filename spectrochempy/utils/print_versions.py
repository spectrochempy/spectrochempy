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
import struct
import subprocess
import sys
from importlib.metadata import metadata
from os import environ

from spectrochempy.utils import optional
from spectrochempy.utils.file import pathclean

__all__ = ["show_versions"]


def show_versions(file=sys.stdout):
    """print the versions of spectrochempy and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """

    print("\nSYSTEM INFO", file=file)
    print("-------------", file=file)

    for key, val in _get_sys_info():
        print(f"- {key:25s} {val}", file=file)
    print(file=file)

    # dependencies
    deps, deps_dev, deps_test = get_package_requirements()

    RENAME = {}
    for li in (deps, deps_dev, deps_test):
        RENAME.update({k: k.replace("-", "_") for k in li if "@" not in k})
    RENAME.update(
        {
            "numpy-quaternion": "quaternion",  # numpy-quaternion is imported as quaternion
            "scikit-learn": "sklearn",
            "pyyaml": "yaml",
            "gitpython": "git",
            "ipython": "IPython",
            "pep8_naming": "pep8ext_naming",
            "pyzmq": "zmq",
        }
    )

    print("\nINSTALLED PACKAGES", file=file)
    print("------------------", file=file)
    ss = "Package requirenents"
    print(f"{ss:28s}Installed version", file=file)

    versions = _find_versions(deps, RENAME)
    for dep, version in versions.items():
        print(f"- {dep:25s} {version}", file=file)

    # dev dependencies
    if deps_dev:
        print("\nDEV PACKAGES", file=file)
        print("------------", file=file)

        versions = _find_versions(deps_dev, RENAME)
        for dep, version in versions.items():
            print(f"- {dep:25s} {version}", file=file)

    if deps_test:
        print("\nTEST PACKAGES", file=file)
        print("------------", file=file)

        versions = _find_versions(deps_test, RENAME)
        for dep, version in versions.items():
            print(f"- {dep:25s} {version}", file=file)
    return


def _get_sys_info():
    """Returns system information as a dict"""
    # copied from XArray

    REPOS = pathclean(__file__).parent.parent.parent

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
                ("LC_ALL", f'{environ.get("LC_ALL", "None")}'),
                ("LANG", f'{environ.get("LANG", "None")}'),
                ("LOCALE", f"{locale.getlocale()}"),
            ]
        )
    except Exception:
        pass

    return blob


def get_package_requirements():
    # Get metadata for spectrochempy
    meta = metadata("spectrochempy")

    # Extract requirements from Requires-Dist
    reqs = meta.get_all("Requires-Dist")
    reqs = [[x.strip().split("@")[0] for x in req.split(";")] for req in reqs]
    deps = [req[0] for req in reqs if len(req) == 1]
    deps_dev = [req[0] for req in reqs if len(req) == 2 and "dev" in req[1]]
    deps_test = [
        req[0]
        for req in reqs
        if len(req) == 2 and "test" in req[1] and req[0] not in deps_dev
    ]
    return deps, deps_dev, deps_test


def _find_versions(deps, RENAME):

    versions = {}
    for dep_ in deps:
        try:
            dep = dep_.split(">")[0]
            dep = dep.split("<")[0]
            dep = dep.split("=")[0]
            dep = RENAME.get(dep, dep)
            try:
                module = optional.import_optional_dependency(dep)
            except Exception:
                print(f"Could not import {dep}")
                continue
            version = optional.get_module_version(module)
            versions[dep_] = version
        except ImportError:

            versions[dep_] = "not found"

    return versions


if __name__ == "__main__":  # pragma: no cover
    show_versions()
