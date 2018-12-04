# -*- coding: utf-8 -*-
#
# ============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ============================================================================

import os
import sys
import io
import json
from pkgutil import walk_packages
from numpy.lib.format import read_array
from numpy.compat import asstr
from traitlets import import_item

__all__ = [

           'list_packages', 'generate_api',

           'make_zipfile', 'ScpFile',

           'unzip'  #tempo

           ]

# ============================================================================
# PACKAGE and API UTILITIES
# ============================================================================

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
        name = _name+'.'+name
    pkgs = sys.modules['spectrochempy.%s' % name]
    api = sys.modules['spectrochempy.%s.api' % name]

    pkgs = list_packages(pkgs)

    __all__ = []

    for pkg in pkgs:
        if pkg.endswith('api'):
            continue
        try:
            pkg = import_item(pkg)
        except:
            raise ImportError(pkg)
        if not hasattr(pkg, '__all__'):
            continue
        a = getattr(pkg, '__all__',[])
        dmethods = getattr(pkg, '__dataset_methods__', [])
        __all__ += a
        for item in a:

            # set general method for the current package API
            setattr(api, item, getattr(pkg, item))

            # some  methods are class method of NDDatasets
            if item in dmethods:
                from spectrochempy.dataset.nddataset import NDDataset
                setattr(NDDataset, item, getattr(pkg, item))

    return __all__

# ============================================================================
# ZIP UTILITIES
# ============================================================================

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
            from spectrochempy.dataset.nddataset import NDDataset
            f = io.BytesIO(self.zip.read(key))
            return NDDataset.load(f)

        elif member and ext in ['.json']:
            return json.loads(asstr(self.zip.read(key)))

        elif member :
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
