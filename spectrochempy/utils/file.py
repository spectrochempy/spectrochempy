# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import os
import sys
import io
import json
from pkgutil import walk_packages
from numpy.lib.format import read_array
from numpy.compat import asstr
from traitlets import import_item

from .qtfiledialogs import opendialog, SaveFileName

__all__ = ['readfilename', 'readdirname', 'savefilename',
           'list_packages', 'generate_api',
           'make_zipfile', 'ScpFile',
           'unzip'  # tempo
           ]


# ======================================================================================================================
# Utility function
# ======================================================================================================================

def readfilename(filename=None, **kwargs):
    """
    returns a list or dictionary of the filenames of existing files, filtered by extensions

    Parameters
    ----------
    filename : `str`, `list` of strings, optional.
        A filename or a list of filenames. If not provided, a dialog box is opened
        to select files in the specified directory or in
        the current directory if not specified.
    directory : `str`, optional.
        The directory where to look at. If not specified, read in
        current directory, or in the datadir if unsuccessful

    filetypes : `list`, optional filter, default=['all files, '.*)'].

    dictionary: `bool`, default=True
        Whether a dictionary or a list should be returns

    Returns
    --------
        list of filenames

    """

    from spectrochempy.core import general_preferences as prefs
    from spectrochempy.api import NO_DISPLAY

    # read input parameters
    directory = kwargs.get("directory", None)
    # alias filetypes and filters as both can be used
    filetypes = kwargs.get("filetypes",
                           kwargs.get("filters", ["all files (*)"]))
    dictionary = kwargs.get("dictionary", True)

    # check passed directory
    if directory:
        if os.path.exists(directory):
            # a valid absolute pathname has been given
            pass
        else:
            # if the directory is not specified we will first look in the current working directory
            # and then in the prefs.datadir
            _directory = os.path.join(os.getcwd(), directory)
            if not os.path.exists(_directory):
                # the directory isn't in the current dir, now try datadir:
                _directory = os.path.join(prefs.datadir, directory)
                if not os.path.exists(_directory):
                    # the directory is definitely not valid... raise an error
                    raise IOError("directory %s doesn't exists!" % directory)
            else:
                directory = _directory

    # now proceed with the filenames
    
    if not filename:
        # test if we are running nbsphinx with this default filename
        filename = os.environ.get('TUTORIAL_FILENAME', None)
        
    if filename:
        _filenames = []
        # make a list, even for a single file name
        filenames = filename
        if not isinstance(filenames, (list, tuple)):
            filenames = list([filenames])
        else:
            filenames = list(filenames)

        # look if all the filename exists either in the specified directory,
        # else in the current directory, and finally in the default preference data directory
        for i, filename in enumerate(filenames):
            if directory:
                _f = os.path.expanduser(os.path.join(directory, filename))
            else:
                _f = filename
                if not os.path.exists(_f):
                    # the filename provided doesn't exists in the specified directory
                    # or the current directory let's try in the default data directory
                    _f = os.path.join(prefs.datadir, filename)
                    if not os.path.exists(_f):
                        raise IOError("Can't find  this filename %s in the specified directory "
                                      "(or in the current one, or in the default data directory %s"
                                      "if directory was not specified " % (filename, prefs.datadir))
            _filenames.append(_f)

        # now we have all the filename with their correct location
        filenames = _filenames
        
    if not filename:
        # open a file dialog
        # currently Scpy use QT (needed for next GUI features)
        filenames = None
        if not directory:
            directory = os.getcwd()

        # We can not do this during full pytest run without blocking the process
        # TODO: use the pytest-qt to solve this problem
        if not NO_DISPLAY:
            filenames = opendialog(single=False,
                                   directory=directory,
                                   caption='select files',
                                   filters=filetypes)
        if not filenames:
            # the dialog has been cancelled or return nothing
            return None

    # now we have either a list of the selected files
    if isinstance(filenames, list):
        if not all(isinstance(elem, str) for elem in filenames):
            raise IOError('one of the list elements is not a filename!')

    # or a single filename
    if isinstance(filenames, str):
        filenames = [filenames]

    # make and return a dictionary
    if dictionary:
        filenames_dict = {}
        for filename in filenames:
            if filename.endswith('.DS_Store'):
                # avoid storing bullshit sometime present in the directory (MacOSX)
                continue
            _, extension = os.path.splitext(filename)
            extension = extension.lower()
            if extension in filenames_dict.keys():
                filenames_dict[extension].append(filename)
            else:
                filenames_dict[extension] = [filename]
        return filenames_dict
    # or just a list
    else:
        return filenames


def readdirname(dirname=None, **kwargs):
    """
    returns a valid directory name

    Parameters
    ----------
    dirname : `str`, optional.
        A directory name. If not provided, a dialog box is opened to select a directory.
    parent_dir : `str`, optional.
        The parent directory where to look at. If not specified, read in the current working directory

    Returns
    --------
        valid directory name
    """

    from spectrochempy.core import general_preferences as prefs
    from spectrochempy.api import NO_DISPLAY

    # Check parent directory
    parent_dir = kwargs.get("parent_dir", None)
    if parent_dir is not None:
        if os.path.isdir(parent_dir):
            pass
        elif prefs.datadir == parent_dir:
            pass
        elif os.path.isdir(os.path.join(prefs.datadir, parent_dir)):
            parent_dir = os.path.join(prefs.datadir, parent_dir)
        else:
            raise ValueError("\"%s\" is not a valid parent directory " % parent_dir)

    if dirname:
        # if a directory name was provided
        # first look if the type is OK
        if not isinstance(dirname, str):
            # well the directory doesn't exist - we cannot go further without
            # correcting this error
            raise TypeError("directory %s should be a string!" % dirname)

        # if a valid parent directory has been provided,
        # checks that parent_dir\\dirname is OK
        if parent_dir is not None:
            if os.path.isdir(os.path.join(parent_dir, dirname)):
                return os.path.join(parent_dir, dirname)
        # if no parent directory: look at current working dir
        elif os.path.isdir(os.path.join(os.getcwd(), dirname)):
            return os.path.join(os.getcwd(), dirname)
        # if no current directory: look at data dir
        elif os.path.isdir(os.path.join(prefs.datadir, dirname)):
            return os.path.join(prefs.datadir, dirname)

        else:
            raise ValueError("\"%s\" is not a valid directory" % dirname)

    if not dirname:
        # open a file dialog
        # currently Scpy use QT (needed for next GUI features)

        if not parent_dir:
            # if no parent directory was specified
            parent_dir = os.getcwd()

        caption = kwargs.get('caption', 'Select folder')

        if not NO_DISPLAY:  # this is for allowing test to continue in the background
            directory = opendialog(single=False,
                                   directory=parent_dir,
                                   caption=caption,
                                   filters='directory')

            return directory


def savefilename(filename: None, directory: None, filters: None):
    """returns a valid filename to save a file

    Parameters
    ----------
    filename : `str`, optional.
        A filename. If not provided, a dialog box is opened to select a file.
    directoryr: `str`, optional.
        The directory where to save the file. If not specified, use the current working directory

    Returns
    --------
        a valid filename to save a file
    """
    from spectrochempy.api import NO_DISPLAY

    # check if directory is specified
    if directory and not os.path.exists(directory):
        raise IOError('Error : Invalid directory given')

    if not directory:
        directory = os.getcwd()

    if filename:
        filename = os.path.join(directory, filename)
    else:
        # no filename was given then open a dialog box
        # currently Scpy use QT (needed for next GUI features)

        # We can not do this during full pytest run without blocking the process
        # TODO: use the pytest-qt to solve this problem

        if not filters:
            filters = "All files (*)"
        if not NO_DISPLAY:
            filename = SaveFileName(parent=None, filters=filters)
        if not filename:
            # if the dialog has been cancelled or return nothing
            return None

    return filename


# ======================================================================================================================
# PACKAGE and API UTILITIES
# ======================================================================================================================

# ............................................................................
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


# ............................................................................
def generate_api(api_path):
    # name of the package
    dirname, name = os.path.split(os.path.split(api_path)[0])
    if not dirname.endswith('spectrochempy'):
        dirname, _name = os.path.split(dirname)
        name = _name + '.' + name
    pkgs = sys.modules['spectrochempy.%s' % name]
    api = sys.modules['spectrochempy.%s.api' % name]

    pkgs = list_packages(pkgs)

    __all__ = []

    for pkg in pkgs:
        if pkg.endswith('api') or "test" in pkg:
            continue
        try:
            pkg = import_item(pkg)
        except:
            pkg = import_item(pkg)
            raise ImportError(pkg)
        if not hasattr(pkg, '__all__'):
            continue
        a = getattr(pkg, '__all__', [])
        dmethods = getattr(pkg, '__dataset_methods__', [])
        __all__ += a
        for item in a:

            # set general method for the current package API
            setattr(api, item, getattr(pkg, item))

            # some  methods are class method of NDDatasets
            if item in dmethods:
                from spectrochempy.core.dataset.nddataset import NDDataset
                setattr(NDDataset, item, getattr(pkg, item))

    return __all__


# ======================================================================================================================
# ZIP UTILITIES
# ======================================================================================================================

# ............................................................................
def make_zipfile(file, **kwargs):
    """
    Create a ZipFile.

    Allows for Zip64 (useful if files are larger than 4 GiB, and the `file`
    argument can accept file or str.
    `kwargs` are passed to the zipfile.ZipFile
    constructor.

    (adapted from numpy)

    """
    import zipfile
    kwargs['allowZip64'] = True
    return zipfile.ZipFile(file, **kwargs)


# ............................................................................
def unzip(source_filename, dest_dir):
    """
    Unzip a zipped file in a directory

    Parameters
    ----------
    source_filename
    dest_dir

    Returns
    -------

    """
    import zipfile
    with zipfile.ZipFile(source_filename) as zf:
        for member in zf.infolist():
            # Path traversal defense copied from
            # http://hg.python.org/cpython/file/tip/Lib/http/server.py#l789
            words = member.filename.split('/')
            path = dest_dir
            for word in words[:-1]:
                drive, word = os.path.splitdrive(word)
                head, word = os.path.split(word)
                if word in (os.curdir, os.pardir, ''): continue
                path = os.path.join(path, word)
            zf.extract(member, path)


class ScpFile(object):
    """
    ScpFile(fid)

    (largely inspired by ``NpzFile`` object in numpy)

    `ScpFile` is used to load files stored in ``.scp`` or ``.pscp``
    format.

    It assumes that files in the archive have a ``.npy`` extension in
    the case of the dataset's ``.scp`` file format) ,  ``.scp``  extension
    in the case of project's ``.pscp`` file format and finally ``pars.json``
    files which contains other information on the structure and  attibutes of
    the saved objects. Other files are ignored.

    """

    def __init__(self, fid):
        """
        Parameters
        ----------
        fid : file or str
            The zipped archive to open. This is either a file-like object
            or a string containing the path to the archive.

        """
        _zip = make_zipfile(fid)

        self.files = _zip.namelist()
        self.zip = _zip

        if hasattr(fid, 'close'):
            self.fid = fid
        else:
            self.fid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Close the file.

        """
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None

    def __del__(self):
        self.close()

    def __getitem__(self, key):

        member = False
        base = None
        ext = None

        if key in self.files:
            member = True
            base, ext = os.path.splitext(key)

        if member and ext in [".npy"]:
            f = self.zip.open(key)
            return read_array(f, allow_pickle=True)

        elif member and ext in ['.scp']:
            from spectrochempy.core.dataset.nddataset import NDDataset
            f = io.BytesIO(self.zip.read(key))
            return NDDataset.load(f)

        elif member and ext in ['.json']:
            return json.loads(asstr(self.zip.read(key)))

        elif member:
            return self.zip.read(key)

        else:
            raise KeyError("%s is not a file in the archive or is not "
                           "allowed" % key)

    def __iter__(self):
        return iter(self.files)

    def items(self):
        """
        Return a list of tuples, with each tuple (filename, array in file).

        """
        return [(f, self[f]) for f in self.files]

    def iteritems(self):
        """Generator that returns tuples (filename, array in file)."""
        for f in self.files:
            yield (f, self[f])

    def keys(self):
        """Return files in the archive with a ``.npy``,``.scp`` or ``.json``
        extension."""
        return self.files

    def iterkeys(self):
        """Return an iterator over the files in the archive."""
        return self.__iter__()

    def __contains__(self, key):
        return self.files.__contains__(key)
