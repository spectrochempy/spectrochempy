"""Utility functions for printing version information."""

import locale
import platform
import struct
import subprocess
import sys
from os import environ

from spectrochempy import optional
from spectrochempy.utils import pathclean


__all__ = ["show_versions"]

REPOS = pathclean(__file__).parent.parent.parent


def get_sys_info():
    """Returns system information as a dict"""
    # copied from XArray

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

    print("\nINSTALLED VERSIONS", file=file)
    print("------------------", file=file)

    for key, val in get_sys_info():
        print(f"{key}: {val}", file=file)
    print(file=file)
    deps = []
    with open(REPOS / "environment.yml", "r") as f:
        start = False
        for dep in f.readlines():
            if "dependencies" in dep and not dep.startswith("#"):
                start = True
            if not start:
                continue
            if dep.strip().startswith("-"):
                dep = dep.strip()[2:].split("<")[0].split(">")[0].split("=")[0]
                if dep != "python":
                    deps.append(dep)
    deps.append("spectrochempy")
    for dep in deps:
        mod = optional.import_optional_dependency(dep, errors="ignore")
        try:
            print(
                f"{dep}: {optional.get_module_version(mod) if mod is not None else None}",
                file=file,
            )
        except ImportError:
            print(f"{dep}: (Can't determine version string)", file=file)


if __name__ == "__main__":
    show_versions()
