# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

from os import environ, curdir, pardir, getcwd
import os.path as opath
import sys
import io
import json
import re, fnmatch
from datetime import datetime
import pickle
import base64
import warnings
from pathlib import Path, WindowsPath, PosixPath
from pkgutil import walk_packages

import numpy as np
from numpy.lib.format import read_array
from numpy.compat import asstr
from traitlets import import_item

from spectrochempy.utils.qtfiledialogs import opendialog, SaveFileName

__all__ = ['get_filename', 'readdirname', 'savefilename', 'pathclean',
           'list_packages', 'generate_api',
           'make_zipfile', 'ScpFile',
           'unzip', 'check_filenames', 'check_filename_to_open', 'check_filename_to_save',
           'json_serialiser', 'json_decoder'
           ]


# ======================================================================================================================
# Utility functions
# ======================================================================================================================
def insensitive_case_glob(pattern):
    def either(c):
        return f'[{c.lower()}{c.upper()}]' if c.isalpha() else c
    return ''.join(map(either, pattern))


def pattern(filetypes):
    regex = r"\*\.*\[*[0-9-]*\]*\w*\**"
    patterns = []
    for ft in filetypes:
        m = re.finditer(regex, ft)
        patterns.extend([insensitive_case_glob(match.group(0)) for match in m])
    return patterns


def pathclean(paths):
    """
    Clean a path or a series of path in order to be compatible with windows and unix-based system.

    Parameters
    ----------
    paths :  str or a list of str
        Path to clean. It may contain windows or conventional python separators.

    Returns
    -------
    out : a pathlib object or a list of pathlib objets
        Cleaned path(s)

    Examples
    --------
    >>> from spectrochempy.utils import pathclean

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

    Due to the escape character \ in Unix, path string should be escaped \\ or the raw-string prefix `r` must be used
    as shown below
    >>> filename = pathclean(r"irdata\\nh4y-activation.spg")
    >>> filename.suffix
    '.spg'
    >>> filename.parent.name
    'irdata'

    >>> from spectrochempy import general_preferences as prefs
    >>> datadir = prefs.datadir
    >>> fullpath = datadir / filename

    """
    from spectrochempy.utils import is_windows

    def _clean(path):
        if isinstance(path, Path):
            path = path.name
        if is_windows():
            path = WindowsPath(path)
        else:
            # some replacement so we can handle window style path on unix
            path = path.strip()
            path = path.replace('\\', '/')
            path = path.replace('\n', '/n')
            path = path.replace('\t', '/t')
            path = path.replace('\b', '/b')
            path = path.replace('\a', '/a')
            path = PosixPath(path)
        return Path(path)

    if paths is not None:
        if isinstance(paths, str):
            return _clean(paths).expanduser()
        elif isinstance(paths, (list, tuple)):
            return [_clean(p).expanduser() if isinstance(p, str) else p for p in paths]

    return paths


def check_filename_to_save(*args, **kwargs):
    pass


def check_filename_to_open(*args, **kwargs):
    """
    Check the args and keywords arg to determine the correct filename

    Returns
    -------
    object, filename

    Examples
    --------
    >>> import spectrochempy as scp

    No filename, the objectype is given explicitely
    >>> check_filename_to_open(objtype=scp.NDDataset) #doctest: +ELLIPSIS
    (<class 'spectrochempy.core.dataset.nddataset.NDDataset'>, ...)

    No filename, an instance is passed as the first argument
    >>> check_filename_to_open(scp.NDDataset()) #doctest: +ELLIPSIS
    (<class 'spectrochempy.core.dataset.nddataset.NDDataset'>, ...)

    Only filename specified (by default objectype will be NDDataset
    >>> check_filename_to_open('irdata/CO@Mo_Al2O3.SPG') #doctest: +ELLIPSIS
    (<class 'spectrochempy.core.dataset.nddataset.NDDataset'>, ...)

    Filename provided as well as the object instance
    >>> check_filename_to_open(scp.NDDataset(), 'irdata/CO@Mo_Al2O3.SPG') #doctest: +ELLIPSIS
    (<class 'spectrochempy.core.dataset.nddataset.NDDataset'>, ...)

    Several Filename provided
    >>> check_filename_to_open(scp.NDDataset(), 'wodger.spg', 'irdata/CO@Mo_Al2O3.SPG') #doctest: +ELLIPSIS
    (<class 'spectrochempy.core.dataset.nddataset.NDDataset'>, ...)

    >>> check_filename_to_open(['wodger.spg', 'irdata/CO@Mo_Al2O3.SPG']) #doctest: +ELLIPSIS
    (<class 'spectrochempy.core.dataset.nddataset.NDDataset'>, ...)


    """
    # filename will be given by a keyword parameter except if the first parameters is already the filename
    import spectrochempy as scp

    # by default returned objtype is NDDataset
    objtype = kwargs.get('objtype', scp.NDDataset)

    filenames = None
    # check if the first argument is an instance of NDDataset, NDPanel, or Project
    args = list(args)
    if args:

        if hasattr(args[0], 'implements') and args[0].implements() in ['NDDataset', 'NDPanel', 'Project']:
            # the first arg is an instance of NDDataset, NDPanel or Project
            objtype = type(args.pop(0))

    filenames = check_filenames(*args, **kwargs)
    if not args and filenames is None:
        # this is propbably due to a cancel action for an open dialog.
        return None

    if not isinstance(filenames, dict):

        # deal with some specific cases
        key = filenames[0].suffix.lower()
        if key[1:].isdigit():
            # probably an opus file
            key = '.opus'

        if len(filenames) > 1:
            # or just a list if there is several files of the same type
            if kwargs.get('dictionary', True):
                # return a dictionary
                return objtype, {
                        key: filenames
                        }
            return objtype, filenames
        else:
            # only one file, return it
            if kwargs.get('dictionary', True):
                # return a dictionary
                return objtype, {
                        key: [filenames[0]]
                        }
            return objtype, filenames[0]
    elif args:
        # args where passed so in this case we have directly byte contents instead of filenames only
        contents = filenames
        return objtype, {
                'frombytes': contents
                }
    else:
        # probably no args (which means that we are coming from a dialog or from a full list of a directory
        return objtype, filenames


def check_filenames(*args, **kwargs):
    filenames = None

    if args:
        if isinstance(args[0], (str, Path)):
            # one or several filenames are passed - make Path objects
            filenames = pathclean(args)
        elif isinstance(args[0], bytes):
            # in this case, one or several byte contents has been passed instead of filenames
            # as filename where not given we passed the 'unnamed' string
            # return a dictionary
            return {pathclean(f'no_name_{i}'): arg for i, arg in enumerate(args)}
        elif isinstance(args[0], list):
            if isinstance(args[0][0], (str, Path)):
                filenames = pathclean(args[0])
            elif isinstance(args[0][0], bytes):
                return {pathclean(f'no_name_{i}'): arg for i, arg in enumerate(args[0])}
        elif isinstance(args[0], dict):
            # return directly the dictionary
            return args[0]

    if not filenames:
        # look into keywords (only the case wher ea real str or pathlib filename is given in handled here
        filenames = kwargs.pop('filename', None)
        filenames = [pathclean(filenames)] if pathclean(filenames) is not None else None

    # Look for content in kwargs
    content = kwargs.pop('content', None)
    if content:
        if not filenames:
            filenames = [pathclean('no_name')]
        return {
                filenames[0]: content
                }

    if filenames:
        filenames_ = []
        for filename in filenames:
            # in which directory ?
            directory = filename.parent
            if directory.resolve() == Path.cwd():
                directory = ''
            kw_directory = pathclean(kwargs.get("directory", None))
            if directory and kw_directory and directory != kw_directory:
                # conflit we do not take into account the kw.
                warnings.warn(
                    'Two differents directory where specified (from args and keywords arg). '
                    'Keyword `directory` will be ignored!')
            elif not directory and kw_directory:
                filename = kw_directory / filename
            # check if the file exists here
            f = filename
            if not directory or str(directory).startswith('.'):
                # search first in the current directory
                directory = Path.cwd()
            f = directory / filename
            if f.exists():
                filename = f
            else:
                from spectrochempy.core import general_preferences as prefs
                directory = pathclean(prefs.datadir)
                f = directory / filename
                if f.exists():
                    filename = f
            filenames_.append(filename)
        filenames = filenames_

    else:
        # no filename specified open a dialog
        filetypes = kwargs.pop('filetypes', ['all files (*)'])
        directory = pathclean(kwargs.get("directory", None))
        filenames = get_filename(directory=directory,
                                 dictionary=True,
                                 filetypes=filetypes)
    return filenames


def get_filename(*filenames, **kwargs):
    """
    returns a list or dictionary of the filenames of existing files, filtered by extensions

    Parameters
    ----------
    filenames : `str` or pathlib object, `tuple` or `list` of strings of pathlib object, optional.
        A filename or a list of filenames.
        If not provided, a dialog box is opened to select files in the current directory if no `directory` is specified)
    directory : `str` or pathlib object, optional.
        The directory where to look at. If not specified, read in
        current directory, or in the datadir if unsuccessful
    filetypes : `list`, optional, default=['all files, '.*)'].
        file type filter
    dictionary : `bool`, optional, default=True
        Whether a dictionary or a list should be returned.
    listdir : bool, default=False
        read all file (possibly limited by `filetypes` in a given `directory`.
    recursive : bool, optional,  default=False.
        Read also subfolders

    Warnings
    --------
    if several filenames are provided in the arguments, they must all reside in the same directory!

    Returns
    --------
    out : list of filenames

    Examples
    --------


    """

    from spectrochempy.core import general_preferences as prefs
    from spectrochempy.api import NO_DISPLAY, NO_DIALOG

    # allowed filetypes
    # -----------------
    # alias filetypes and filters as both can be used
    filetypes = kwargs.get("filetypes", kwargs.get("filters", ["all files (*)"]))

    # filenames
    # ---------
    if len(filenames) == 1 and isinstance(filenames[0], (list, tuple)):
        filenames = filenames[0]

    directory = None
    if len(filenames) == 1 and filenames[0].endswith('/'):
        # this specify a directory not a filename
        directory = pathclean(filenames[0])
        filenames = None
    else:
        filenames = pathclean(list(filenames))

    # directory
    # ---------
    if directory is None:
        directory = pathclean(kwargs.get("directory", None))

    if directory is not None:
        if filenames:
            # prepend to the filename (incompatibility between filename and directory specification
            # will result to a error
            filenames = [directory / filename for filename in filenames]
        else:
            directory = readdirname(directory)

    # check the parent directory
    # all filenames must reside in the same directory
    if filenames:
        parents = set()
        for f in filenames:
            parents.add(f.parent)
        if len(parents) > 1:
            raise ValueError('filenames provided have not the same parent directory. '
                             'This is not accepted by the readfilename function.')

        # use readdirname to complete eventual missing part of the absolute path
        directory = readdirname(parents.pop())

        filenames = [filename.name for filename in filenames]

    # now proceed with the filenames
    if filenames:

        # look if all the filename exists either in the specified directory,
        # else in the current directory, and finally in the default preference data directory
        temp = []
        for i, filename in enumerate(filenames):
            if not (directory / filename).exists():
                # the filename provided doesn't exists in the working directory
                # try in the data directory
                directory = pathclean(prefs.datadir)
                if not (directory / filename).exists():
                    raise IOError(f"Can't find  this filename {filename}")
            temp.append(directory / filename)

        # now we have checked all the filename with their correct location
        filenames = temp

    else:
        # no filenames:
        # open a file dialog
        # except if a directory is specified or listdir is True.
        # currently Scpy use QT (needed for next GUI features)

        listdir = kwargs.get('listdir', directory is not None)

        if not listdir:
            # we open a dialogue to select one or several files manually
            if not (NO_DISPLAY or NO_DIALOG):

                filenames = opendialog(single=False,
                                       directory=directory,
                                       caption='select files',
                                       filters=filetypes)
            elif environ.get('TEST_FILE', None) is not None:
                # happen for testing
                filenames = [pathclean(environ.get('TEST_FILE'))]

        else:
            # automatic reading of the whole directory
            filenames = []
            for pat in pattern(filetypes):
                filenames.extend(list(directory.glob(pat)))
            filenames = pathclean(filenames)

        if not filenames:
            # the dialog has been cancelled or return nothing
            return None

    # now we have either a list of the selected files
    if isinstance(filenames, list):
        if not all(isinstance(elem, Path) for elem in filenames):
            raise IOError('one of the list elements is not a filename!')

    # or a single filename
    if isinstance(filenames, Path):
        filenames = [filenames]

    filenames = pathclean(filenames)
    for filename in filenames[:]:
        if filename.name.endswith('.DS_Store'):
            # sometime present in the directory (MacOSX)
            filenames.remove(filename)

    dictionary = kwargs.get("dictionary", True)
    if dictionary:
        # make and return a dictionary
        filenames_dict = {}
        for filename in filenames:
            extension = filename.suffix.lower()
            if extension[1:].isdigit():
                # probably an opus file
                extension = '.opus'
            if extension in filenames_dict.keys():
                filenames_dict[extension].append(filename)
            else:
                filenames_dict[extension] = [filename]
        return filenames_dict
    else:
        return filenames


def readdirname(dirname):
    """
    returns a valid directory name

    Parameters
    ----------
    dirname : `str`, optional.
        A directory name. If not provided, a dialog box is opened to select a directory.

    Returns
    --------
        valid directory name
    """

    from spectrochempy.core import general_preferences as prefs
    from spectrochempy.api import NO_DISPLAY

    data_dir = pathclean(prefs.datadir)
    working_dir = Path.cwd()

    dirname = pathclean(dirname)

    if dirname:
        if dirname.is_dir():
            # nothing else to do
            return dirname

        elif (working_dir / dirname).is_dir():
            # if no parent directory: look at current working dir
            return working_dir / dirname

        elif (data_dir / dirname).is_dir():
            return data_dir / dirname

        else:
            # raise ValueError(f'"{dirname}" is not a valid directory')
            warnings.warn(f'"{dirname}" is not a valid directory')
            return None

    else:
        # open a file dialog
        dirname = data_dir
        if not NO_DISPLAY:  # this is for allowing test to continue in the background
            dirname = opendialog(single=False,
                                 directory=working_dir,
                                 caption='Select directory',
                                 filters='directory')

        return pathclean(dirname)


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
    if directory and not opath.exists(directory):
        raise IOError('Error : Invalid directory given')

    if not directory:
        directory = getcwd()

    if filename:
        filename = opath.join(directory, filename)
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

    dirname, name = opath.split(opath.split(api_path)[0])
    if not dirname.endswith('spectrochempy'):
        dirname, _name = opath.split(dirname)
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
        except Exception:
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
                from spectrochempy.core.dataset.nddataset import NDIO
                setattr(NDIO, item, getattr(pkg, item))

    return __all__


# ======================================================================================================================
# JSON UTILITIES
# ======================================================================================================================

def json_serialiser(byte_obj):
    if isinstance(byte_obj, datetime):
        return {
                "isoformat": byte_obj.isoformat(),
                "__class__": str(byte_obj.__class__)
                }
    if isinstance(byte_obj, np.ndarray):
        # return {"ndarray":byte_obj.tolist(), "dtype": byte_obj.dtype.name}
        return {
                "serialized": base64.b64encode(pickle.dumps(byte_obj)).decode(),
                "__class__" : str(byte_obj.__class__)
                }
    raise ValueError('No encoding handler for data type ' + type(byte_obj))


def json_decoder(dic):
    if "__class__" in dic:
        if dic["__class__"] == str(datetime):
            return datetime.fromisoformat(dic["isoformat"])
        if dic["__class__"] == str(np.ndarray):
            return pickle.loads(base64.b64decode(dic['serialized']))
        raise TypeError("numpy array or datetime expected.")
    return dic


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
                drive, word = opath.splitdrive(word)
                head, word = opath.split(word)
                if word in (curdir, pardir, ''):
                    continue
                path = opath.join(path, word)
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
            base, ext = opath.splitext(key)

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

# EOF
