# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

__all__ = [
    "get_user_and_node",
    "get_user",
    "get_node",
    "is_kernel",
    "sh",
    "is_windows",
]

import getpass
import platform
import sys
from subprocess import run, PIPE, STDOUT


def is_windows():
    win = "Windows" in platform.platform()
    return win


def get_user():
    return getpass.getuser()


def get_node():
    return platform.node()


def get_user_and_node():
    return f"{get_user()}@{get_node()}"


def is_kernel():
    """
    Check if we are running from IPython.
    """
    # from http://stackoverflow.com
    # /questions/34091701/determine-if-were-in-an-ipython-notebook-session
    if "IPython" not in sys.modules:
        # IPython hasn't been imported
        return False  # pragma: no cover
    from IPython import get_ipython  # pragma: no cover

    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), "kernel", None) is not None  # pragma: no cover


class _ExecCommand:
    def __init__(self, command):
        """
        Parameters
        ----------
        command: shell command to execute
        """
        self.commands = [command]

    def __call__(self, *args, **kwargs):

        self.commands.extend(args)

        silent = kwargs.pop("silent", False)
        proc = run(
            self.commands, text=True, stdout=PIPE, stderr=STDOUT
        )  # capture_output=True)

        # TODO: handle error codes
        if not silent and proc.stdout:
            print(proc.stdout)
        return proc.stdout


# noinspection PyPep8Naming
class sh(object):
    """
    Utility to run subprocess run command as if they were functions.
    """

    def __getattr__(self, command):

        return _ExecCommand(command)

    def __call__(self, script, silent=False):
        # use to run shell script

        proc = run(script, text=True, shell=True, stdout=PIPE, stderr=STDOUT)

        if not silent:
            print(proc.stdout)
        return proc.stdout


sh = sh()
