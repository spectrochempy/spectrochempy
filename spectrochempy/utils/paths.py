# -*- coding: utf-8 -*-

# ======================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory
# ======================================================================================
"""
This module defines utilities related to paths manipulation.
"""

from pathlib import Path, PosixPath, WindowsPath


def pathclean(pths):
    """
    Clean a path or a series of path.

    This cleaning is done in order to be compatible with windows and Unix-based system.

    Parameters
    ----------
    pths :  str or a list of str
        Path to clean. It may contain windows or conventional python separators.

    Returns
    -------
    out : a pathlib object or a list of pathlib objects
        Cleaned path(s)

    Examples
    --------
    >>> from spectrochempy.utils.paths import pathclean

    Using unix/mac way to write paths
    >>> filename = pathclean('irdata/nh4y-activation.spg')
    >>> filename.suffix
    '.spg'
    >>> filename.parent.name
    'irdata'

    or Windows
    >>> filename = pathclean("irdata\\\\nh4y-activation.spg")
    >>> filename.parent.name
    'irdata'

    Due to the escape character \\ in Unix, path string should be escaped \\\\
    or the raw-string prefix `r` must be used
    as shown below
    >>> filename = pathclean(r"irdata\\nh4y-activation.spg")
    >>> filename.suffix
    '.spg'
    >>> filename.parent.name
    'irdata'
    """
    import platform

    def is_windows():
        win = "Windows" in platform.platform()
        return win

    def _clean(pth):
        if isinstance(pth, (Path, PosixPath, WindowsPath)):
            pth = pth.name
        if is_windows():
            pth = WindowsPath(pth)  # pragma: no cover
        else:  # some replacement so we can handle window style path on unix
            pth = pth.strip()
            pth = pth.replace("\\", "/")
            pth = pth.replace("\n", "/n")
            pth = pth.replace("\t", "/t")
            pth = pth.replace("\b", "/b")
            pth = pth.replace("\a", "/a")
            pth = PosixPath(pth)
        return Path(pth)

    if pths is not None:
        if isinstance(pths, (str, Path, PosixPath, WindowsPath)):
            path = str(pths)
            return _clean(path).expanduser()
        if isinstance(pths, (list, tuple)):
            return [_clean(p).expanduser() if isinstance(p, str) else p for p in pths]

    return pths
