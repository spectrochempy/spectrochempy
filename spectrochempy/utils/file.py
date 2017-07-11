#------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------

""" This module contains functions to determine where configuration and
data/cache files used by Spectrochempy should be placed."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['File',
           'get_log_dir',
           'get_config_dir',
           'get_cache_dir',
           'set_temp_config',
           'set_temp_cache',
           'get_pkg_data_dir',
           'get_pkg_data_filename',
          ]

# Standard/built-in imports.
import mimetypes
import os
import shutil
import stat
import sys
import six

from traitlets import Bool, HasTraits, Instance, List, Unicode

#from .decorators import wraps
from .introspect import find_current_module, resolve_name

class File(HasTraits):
    """ A representation of files and folders in a file system. """

    # The path name of this file/folder.
    path = Unicode


    def __init__(self, path, **traits):
        """ Creates a new representation of the specified path. """

        super(File, self).__init__(path=path, **traits)

        return

    def __cmp__(self, other):
        """ Comparison operators. """
        if isinstance(other, File):
            return cmp(self.path, other.path)

        return 1

    def __str__(self):
        """ Returns an 'informal' string representation of the object. """

        return 'File(%s)' % self.path


    #### Properties ###########################################################

    # The absolute path name of this file/folder.

    @property
    def absolute_path(self):
        """ Returns the absolute path of this file/folder. """

        return os.path.abspath(self.path)

    @property
    def children(self):
        """ Returns the folder's children.

        Returns None if the path does not exist or is not a folder.

        """

        if self.is_folder:
            children = []
            for name in os.listdir(self.path):
                children.append(File(os.path.join(self.path, name)))

        else:
            children = None

        return children

    @property
    def exists(self):
        """ Returns True if the file exists, otherwise False. """

        return os.path.exists(self.path)

    @property
    def ext(self):
        """ Returns the file extension. """

        name, ext = os.path.splitext(self.path)

        return ext

    @property
    def is_file(self):
        """ Returns True if the path exists and is a file. """

        return self.exists and os.path.isfile(self.path)

    @property
    def is_folder(self):
        """ Returns True if the path exists and is a folder. """

        return self.exists and os.path.isdir(self.path)

    @property
    def is_package(self):
        """ Returns True if the path exists and is a Python package. """

        return self.is_folder and '__init__.py' in os.listdir(self.path)

    @property
    def is_readonly(self):
        """ Returns True if the file/folder is readonly, otherwise False. """

        # If the File object is a folder, os.access cannot be used because it
        # returns True for both read-only and writable folders on Windows
        # systems.
        if self.is_folder:

            # Mask for the write-permission bits on the folder. If these bits
            # are set to zero, the folder is read-only.
            WRITE_MASK = 0x92
            permissions = os.stat(self.path)[0]

            if permissions & WRITE_MASK == 0:
                readonly = True
            else:
                readonly = False

        elif self.is_file:
            readonly = not os.access(self.path, os.W_OK)

        else:
            readonly = False

        return readonly

    @property
    def mime_type(self):
        """ Returns the mime-type of this file/folder. """

        mime_type, encoding = mimetypes.guess_type(self.path)
        if mime_type is None:
            mime_type = "content/unknown"

        return mime_type

    @property
    def name(self):
        """ Returns the last component of the path without the extension. """

        basename = os.path.basename(self.path)

        name, ext = os.path.splitext(basename)

        return name

    @property
    def parent(self):
        """ Returns the parent of this file/folder. """

        return File(os.path.dirname(self.path))

    @property
    def url(self):
        """ Returns the path as a URL. """

        # Unicodeip out the leading slash on POSIX systems.
        return 'file:///%s' % self.absolute_path.lstrip('/')

    #### Methods ##############################################################

    def copy(self, destination):
        """ Copies this file/folder. """

        # Allow the destination to be a string.
        if not isinstance(destination, File):
            destination = File(destination)

        if self.is_folder:
            shutil.copytree(self.path, destination.path)

        elif self.is_file:
            shutil.copyfile(self.path, destination.path)

        return

    def create_file(self, contents=''):
        """ Creates a file at this path. """

        if self.exists:
            raise ValueError("file %s already exists" % self.path)

        f = open(self.path, 'w')
        f.write(contents)
        f.close()

        return

    def create_folder(self):
        """ Creates a folder at this path.

        All intermediate folders MUST already exist.

        """

        if self.exists:
            raise ValueError("folder %s already exists" % self.path)

        os.mkdir(self.path)

        return

    def create_folders(self):
        """ Creates a folder at this path.

        This will attempt to create any missing intermediate folders.

        """

        if self.exists:
            raise ValueError("folder %s already exists" % self.path)

        os.makedirs(self.path)

        return

    def create_package(self):
        """ Creates a package at this path.

        All intermediate folders/packages MUST already exist.

        """

        if self.exists:
            raise ValueError("package %s already exists" % self.path)

        os.mkdir(self.path)

        # Create the '__init__.py' file that actually turns the folder into a
        # package!
        init = File(os.path.join(self.path, '__init__.py'))
        init.create_file()

        return

    def delete(self):
        """ Deletes this file/folder.

        Does nothing if the file/folder does not exist.

        """

        if self.is_folder:
            # Try to make sure that everything in the folder is writeable.
            self.make_writeable()

            # Delete it!
            shutil.rmtree(self.path)

        elif self.is_file:
            # Try to make sure that the file is writeable.
            self.make_writeable()

            # Delete it!
            os.remove(self.path)

        return

    def make_writeable(self):
        """ Attempt to make the file/folder writeable. """

        if self.is_folder:
            # Try to make sure that everything in the folder is writeable
            # (i.e., can be deleted!).  This comes in especially handy when
            # deleting '.svn' directories.
            for path, dirnames, filenames in os.walk(self.path):
                for name in dirnames + filenames:
                    filename = os.path.join(path, name)
                    if not os.access(filename, os.W_OK):
                        os.chmod(filename, stat.S_IWUSR)

        elif self.is_file:
            # Try to make sure that the file is writeable (i.e., can be
            # deleted!).
            if not os.access(self.path, os.W_OK):
                os.chmod(self.path, stat.S_IWUSR)

        return

    def move(self, destination):
        """ Moves this file/folder. """

        # Allow the destination to be a string.
        if not isinstance(destination, File):
            destination = File(destination)

        # Try to make sure that everything in the directory is writeable.
        self.make_writeable()

        # Move it!
        shutil.move(self.path, destination.path)

        return

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


    # this is used below to make fix up encoding issues that sometimes crop up
    # in py2.x but not in py3.x
    if six.PY2:
        decodepath = lambda pth: pth.decode(sys.getfilesystemencoding())
    elif six.PY3:
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
                from six.moves import winreg as wreg
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

    # If using set_temp_config, that overrides all
    if set_temp_config._temp_path is not None:
        xch = set_temp_config._temp_path
        config_path = os.path.join(xch, 'spectrochempy')
        if not os.path.exists(config_path):
            os.mkdir(config_path)
        return os.path.abspath(config_path)

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

    # If using set_temp_config, that overrides all
    if set_temp_config._temp_path is not None:
        xch = set_temp_config._temp_path
        log_path = os.path.join(xch, 'spectrochempy')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        return os.path.abspath(log_path)

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


def get_cache_dir():
    """
    Determines the SpectroChemPy cache directory name and creates the directory if it
    doesn't exist.

    This directory is typically ``$HOME/.spectrochempy/cache``, but if the
    XDG_CACHE_HOME environment variable is set and the
    ``$XDG_CACHE_HOME/spectrochempy`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    Returns
    -------
    cachedir : str
        The absolute path to the cache directory.

    """

    # symlink will be set to this if the directory is created
    linkto = None

    # If using set_temp_cache, that overrides all
    if set_temp_cache._temp_path is not None:
        xch = set_temp_cache._temp_path
        cache_path = os.path.join(xch, 'spectrochempy')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return os.path.abspath(cache_path)

    # first look for XDG_CACHE_HOME
    xch = os.environ.get('XDG_CACHE_HOME')

    if xch is not None and os.path.exists(xch):
        xchpth = os.path.join(xch, 'spectrochempy')
        if not os.path.islink(xchpth):
            if os.path.exists(xchpth):
                return os.path.abspath(xchpth)
            else:
                linkto = xchpth

    return os.path.abspath(_find_or_create_spectrochempy_dir('cache', linkto))


class _SetTempPath(object):
    _temp_path = None
    _default_path_getter = None

    def __init__(self, path=None, delete=False):
        if path is not None:
            path = os.path.abspath(path)

        self._path = path
        self._delete = delete
        self._prev_path = self.__class__._temp_path

    def __enter__(self):
        self.__class__._temp_path = self._path
        return self._default_path_getter()

    def __exit__(self, *args):
        self.__class__._temp_path = self._prev_path

        if self._delete and self._path is not None:
            shutil.rmtree(self._path)

    def __call__(self, func):
        """Implements use as a decorator."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                func(*args, **kwargs)

        return wrapper


class set_temp_config(_SetTempPath):
    """
    Context manager to set a temporary path for the SpectroChemPy config, primarily
    for use with testing.

    If the path set by this context manager does not already exist it will be
    created, if possible.

    This may also be used as a decorator on a function to set the config path
    just within that function.

    Parameters
    ----------

    path : str, optional
        The directory (which must exist) in which to find the SpectroChemPy config
        files, or create them if they do not already exist.  If None, this
        restores the config path to the user's default config path as returned
        by `get_config_dir` as though this context manager were not in effect
        (this is useful for testing).  In this case the ``delete`` argument is
        always ignored.

    delete : bool, optional
        If True, cleans up the temporary directory after exiting the temp
        context (default: False).
    """

    _default_path_getter = staticmethod(get_config_dir)

    def __enter__(self):
        # Special case for the config case, where we need to reset all the
        # cached config objects
        #from .configuration import _cfgobjs

        path = super(set_temp_config, self).__enter__()
        #_cfgobjs.clear()
        return path

    def __exit__(self, *args):
        #from .configuration import _cfgobjs

        super(set_temp_config, self).__exit__(*args)
        #_cfgobjs.clear()


class set_temp_cache(_SetTempPath):
    """
    Context manager to set a temporary path for the SpectroChemPy download cache,
    primarily for use with testing (though there may be other applications
    for setting a different cache directory, for example to switch to a cache
    dedicated to large files).

    If the path set by this context manager does not already exist it will be
    created, if possible.

    This may also be used as a decorator on a function to set the cache path
    just within that function.

    Parameters
    ----------

    path : str
        The directory (which must exist) in which to find the SpectroChemPy cache
        files, or create them if they do not already exist.  If None, this
        restores the cache path to the user's default cache path as returned
        by `get_cache_dir` as though this context manager were not in effect
        (this is useful for testing).  In this case the ``delete`` argument is
        always ignored.

    delete : bool, optional
        If True, cleans up the temporary directory after exiting the temp
        context (default: False).
    """

    _default_path_getter = staticmethod(get_cache_dir)


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

#### EOF ######################################################################
