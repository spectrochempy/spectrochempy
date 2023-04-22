# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
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
from os import environ
from pathlib import Path

__all__ = ["show_versions"]


def _get_sys_info():
    """Returns system information as a dict"""
    # copied from XArray
    from spectrochempy.utils.file import pathclean

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


def show_versions(file=sys.stdout):
    """print the versions of spectrochempy and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    from spectrochempy.utils import optional

    print("\nINSTALLED VERSIONS", file=file)
    print("------------------", file=file)

    for key, val in _get_sys_info():
        print(f"{key}: {val}", file=file)
    print(file=file)

    # dependencies
    deps = []
    reqs = Path(__file__).parent.parent.parent / "requirements" / "requirements_dev.txt"
    reqs = reqs.read_text().split("\n")
    for req in reqs:
        req = req.strip()
        if req == "" or req.startswith("#"):
            continue
        deps.append(req)

    for dep in deps:
        mod = optional.import_optional_dependency(dep, errors="ignore")
        try:
            print(
                f"{dep}: "
                f"{optional.get_module_version(mod) if mod is not None else None}",
                file=file,
            )
        except ImportError:
            print(f"{dep}: (Can't determine version string)", file=file)


if __name__ == "__main__":  # pragma: no cover
    show_versions()
