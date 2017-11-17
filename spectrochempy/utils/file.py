# ------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#
# Author: Enthought, Inc.
# Description: <Enthought IO package component>
# ------------------------------------------------------------------------------

""" This module contains functions to determine where configuration and
data files used by Spectrochempy should be placed.

"""

__all__ = ['get_log_dir',
           'get_config_dir',
           'get_pkg_data_dir',
           'get_pkg_data_filename',
           'list_packages',
           ]

_methods = ['get_log_dir',
            'get_config_dir',
            'get_pkg_data_dir',
            'get_pkg_data_filename',
            'list_packages',
            ]

import os
import sys
from pkgutil import walk_packages

from .introspect import find_current_module, resolve_name


def _find_home():
    """ Locates and return the home directory (or best approximation) on this
    system.

    Raises
    ------
    OSError
        If the home directory cannot be located - usually means you are running
        SpectroChemPy on some obscure platform that doesn't have standard home
        directories.
    """

    decodepath = lambda pth: pth

    # First find the home directory - this is inspired by the scheme ipython
    # uses to identify "home"
    if os.name == 'posix':
        # Linux, Unix, AIX, OS X
        if 'HOME' in os.environ:
            homedir = decodepath(os.environ['HOME'])
        else:
            raise OSError('Could not find unix home directory to search for '
                          'spectrochempy config dir')
    elif os.name == 'nt':  # This is for all modern Windows (NT or after)
        if 'MSYSTEM' in os.environ and os.environ.get('HOME'):
            # Likely using an msys shell; use whatever it is using for its
            # $HOME directory
            homedir = decodepath(os.environ['HOME'])
        # Next try for a network home
        elif 'HOMESHARE' in os.environ:
            homedir = decodepath(os.environ['HOMESHARE'])
        # See if there's a local home
        elif 'HOMEDRIVE' in os.environ and 'HOMEPATH' in os.environ:
            homedir = os.path.join(os.environ['HOMEDRIVE'],
                                   os.environ['HOMEPATH'])
            homedir = decodepath(homedir)
        # Maybe a user profile?
        elif 'USERPROFILE' in os.environ:
            homedir = decodepath(os.path.join(os.environ['USERPROFILE']))
        else:
            try:
                import winreg as wreg
                shell_folders = r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
                key = wreg.OpenKey(wreg.HKEY_CURRENT_USER, shell_folders)

                homedir = wreg.QueryValueEx(key, 'Personal')[0]
                homedir = decodepath(homedir)
                key.Close()
            except:
                # As a final possible resort, see if HOME is present
                if 'HOME' in os.environ:
                    homedir = decodepath(os.environ['HOME'])
                else:
                    raise OSError('Could not find windows home directory to '
                                  'search for spectrochempy config dir')
    else:
        # for other platforms, try HOME, although it probably isn't there
        if 'HOME' in os.environ:
            homedir = decodepath(os.environ['HOME'])
        else:
            raise OSError('Could not find a home directory to search for '
                          'spectrochempy config dir - are you on an unspported '
                          'platform?')
    return homedir


def get_config_dir(create=True):
    """
    Determines the SpectroChemPy configuration directory name and creates the
    directory if it doesn't exist.

    This directory is typically ``$HOME/.spectrochempy/config``, but if the
    XDG_CONFIG_HOME environment variable is set and the
    ``$XDG_CONFIG_HOME/spectrochempy`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    Returns
    -------
    configdir : str
        The absolute path to the configuration directory.

    """

    # symlink will be set to this if the directory is created
    linkto = None

    # first look for XDG_CONFIG_HOME
    xch = os.environ.get('XDG_CONFIG_HOME')

    if xch is not None and os.path.exists(xch):
        xchpth = os.path.join(xch, 'spectrochempy')
        if not os.path.islink(xchpth):
            if os.path.exists(xchpth):
                return os.path.abspath(xchpth)
            else:
                linkto = xchpth
    return os.path.abspath(_find_or_create_spectrochempy_dir('config', linkto))


def get_log_dir(create=True):
    """
    Determines the SpectroChemPy logging directory name and creates the
    directory if it doesn't exist.

    This directory is typically ``$HOME/.spectrochempy/log``, but if the
    XDG_LOG_HOME environment variable is set and the
    ``$XDG_LOG_HOME/spectrochempy`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    Returns
    -------
    logdir : str
        The absolute path to the looging directory.

    """

    # symlink will be set to this if the directory is created
    linkto = None

    # first look for XDG_LOG_HOME
    xch = os.environ.get('XDG_LOG_HOME')

    if xch is not None and os.path.exists(xch):
        xchpth = os.path.join(xch, 'spectrochempy')
        if not os.path.islink(xchpth):
            if os.path.exists(xchpth):
                return os.path.abspath(xchpth)
            else:
                linkto = xchpth
    return os.path.abspath(_find_or_create_spectrochempy_dir('log', linkto))


def _find_or_create_spectrochempy_dir(dirnm, linkto):
    innerdir = os.path.join(_find_home(), '.spectrochempy')
    maindir = os.path.join(_find_home(), '.spectrochempy', dirnm)

    if not os.path.exists(maindir):
        # first create .spectrochempy dir if needed
        if not os.path.exists(innerdir):
            try:
                os.mkdir(innerdir)
            except OSError:
                if not os.path.isdir(innerdir):
                    raise
        elif not os.path.isdir(innerdir):
            msg = 'Intended SpectroChemPy directory {0} is actually a file.'
            raise IOError(msg.format(innerdir))

        try:
            os.mkdir(maindir)
        except OSError:
            if not os.path.isdir(maindir):
                raise

        if (not sys.platform.startswith('win') and
                    linkto is not None and
                not os.path.exists(linkto)):
            os.symlink(maindir, linkto)

    elif not os.path.isdir(maindir):
        msg = 'Intended SpectroChemPy {0} directory {1} is actually a file.'
        raise IOError(msg.format(dirnm, maindir))

    return os.path.abspath(maindir)


def _is_inside(path, parent_path):
    # We have to try realpath too to avoid issues with symlinks, but we leave
    # abspath because some systems like debian have the absolute path (with no
    # symlinks followed) match, but the real directories in different
    # locations, so need to try both cases.
    return os.path.abspath(path).startswith(os.path.abspath(parent_path)) \
           or os.path.realpath(path).startswith(os.path.realpath(parent_path))


def _find_pkg_data_path(data_name, package=None):
    """
    Look for data in the source-included data directories and return the
    path.
    """

    if package is None:
        module = find_current_module(1, True)

        if module is None:
            # not called from inside an astropy package.  So just pass name
            # through
            return data_name

        if not hasattr(module, '__package__') or not module.__package__:
            # The __package__ attribute may be missing or set to None; see
            # PEP-366, also astropy issue #1256
            if '.' in module.__name__:
                package = module.__name__.rpartition('.')[0]
            else:
                package = module.__name__
        else:
            package = module.__package__
    else:
        module = resolve_name(package)

    rootpkgname = package.partition('.')[0]

    rootpkg = resolve_name(rootpkgname)

    module_path = os.path.dirname(module.__file__)
    path = os.path.join(module_path, data_name)

    root_dir = os.path.dirname(rootpkg.__file__)
    assert _is_inside(path, root_dir), \
        ("attempted to get a local data file outside "
         "of the " + rootpkgname + " tree")

    return path


def get_pkg_data_filename(data_name, package=None):
    """
    Retrieves a data filename

    Parameters
    ----------
    data_name : str

    package : str, optional
        If specified, look for a file relative to the given package, rather
        than the default of looking relative to the calling module's package.

    Raises
    ------
    IOError
        If problems occur writing or reading a local file.

    Returns
    -------
    filename : str
        A file path on the local file system corresponding to the data
        requested in ``data_name``.


    """

    data_name = os.path.normpath(data_name)

    datafn = _find_pkg_data_path(data_name, package=package)
    if os.path.isdir(datafn):
        raise IOError("Tried to access a data file that's actually "
                      "a package data directory")
    return datafn


def get_pkg_data_dir(data_name, package=None):
    """
    Retrieves a data directory

    Parameters
    ----------
    data_name : str

    package : str, optional
        If specified, look for a directory relative to the given package, rather
        than the default of looking relative to the calling module's package.

    Returns
    -------
    filename : str
        A file path on the local file system corresponding to the data directory
        requested in ``data_name``.


    """

    data_name = os.path.normpath(data_name)

    datadir = _find_pkg_data_path(data_name, package=package)

    if not os.path.isdir(datadir):
        return os.path.dirname(datadir)

    return datadir


def list_packages(package):
    """Return a list of the names of a package and its subpackages.

    This only works if the package has a :attr:`__path__` attribute, which is
    not the case for some (all?) of the built-in packages.
    """
    # Based on response at
    # http://stackoverflow.com/questions/1707709

    names = [package.__name__]
    for __, name, __ in walk_packages(package.__path__,
                                      prefix=package.__name__ + '.',
                                      onerror=lambda x: None):
        names.append(name)

    return names

#### EOF ######################################################################
