# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S602, S603

import getpass
import platform
import shlex
import sys
from subprocess import PIPE
from subprocess import STDOUT
from subprocess import run


def get_user():
    return getpass.getuser()


def get_node():
    return platform.node()


def get_user_and_node():
    return f"{get_user()}@{get_node()}"


def is_kernel():
    """Check if we are running from IPython."""
    # from http://stackoverflow.com
    # /questions/34091701/determine-if-were-in-an-ipython-notebook-session
    if "IPython" not in sys.modules:
        # IPython hasn't been imported
        return False  # pragma: no cover
    from IPython import get_ipython  # pragma: no cover

    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), "kernel", None) is not None  # pragma: no cover


class _ExecCommand:
    # """
    # Parameters
    # ----------
    # command : `str`
    #     shell command to execute
    # """

    def __init__(self, command):
        self.commands = [command]

    def __call__(self, *args, **kwargs):
        args = list(args)
        args[-1] = str(args[-1])  # convert Path to str
        self.commands.extend(args)

        silent = kwargs.pop("silent", False)
        safe_command = shlex.split(" ".join(self.commands))
        proc = run(  # noqa: S603
            safe_command,
            text=True,
            stdout=PIPE,
            stderr=STDOUT,
            check=False,
        )  # capture_output=True)

        # TODO: handle error codes
        if not silent and proc.stdout:
            print(proc.stdout)  # noqa: T201
        return proc.stdout


# noinspection PyPep8Naming
class sh:
    """Utility to run subprocess run command as if they were functions."""

    def __getattr__(self, command):
        return _ExecCommand(command)

    def __call__(self, script, silent=False):
        # use to run shell script
        safe_script = shlex.split(script)
        proc = run(  # noqa: S603
            safe_script, text=True, stdout=PIPE, stderr=STDOUT, check=False
        )

        if not silent:
            print(proc.stdout)  # noqa: T201
        return proc.stdout


sh = sh()
