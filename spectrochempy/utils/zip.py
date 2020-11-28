# -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#

import io
import json
from os import curdir, pardir
import os.path as opath
from collections.abc import Mapping

from numpy.lib.format import read_array, MAGIC_PREFIX
from numpy.compat import asstr

__all__ = ['make_zipfile', 'unzip', 'ScpFile']


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

class ScpFile(Mapping):
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

    Attributes
    ----------
    files : list of str
        List of all files in the archive with a ``.npy`` extension.
    zip : ZipFile instance
        The ZipFile object initialized with the zipped archive.

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

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

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


    # def items(self):
    #     """
    #     Return a list of tuples, with each tuple (filename, array in file).
    #
    #     """
    #     return [(f, self[f]) for f in self.files]
    #
    # def keys(self):
    #     """Return files in the archive with a ``.npy``,``.scp`` or ``.json``
    #     extension."""
    #     return self.files

    def __contains__(self, key):
        return self.files.__contains__(key)

# ======================================================================================================================
if __name__ == '__main__':
    pass

# EOF
