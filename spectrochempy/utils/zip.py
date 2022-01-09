# -*- coding: utf-8 -*-

#  =====================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
#  =====================================================================================================================

import os
import json
from collections.abc import Mapping
from numpy.lib.format import read_array

from spectrochempy.utils import json_decoder

__all__ = ["make_zipfile", "ScpFile"]


# ==============================================================================
# ZIP UTILITIES
# ==============================================================================

# ............................................................................
def make_zipfile(file, **kwargs):
    """
    Create a ZipFile.

    Allows for Zip64 (useful if files are larger than 4 GiB)
    (adapted from numpy)

    Parameters
    ----------
    file:  file or str
        The file to be zipped.
    **kwargs: Optional keyword arguments
        They are passed to the zipfile.ZipFile constructor.

    Returns
    -------
    zipfile
    """
    import zipfile

    kwargs["allowZip64"] = True
    return zipfile.ZipFile(file, **kwargs)


class ScpFile(Mapping):  # lgtm[py/missing-equals]
    """
    ScpFile(fid).

    (largely inspired by ``NpzFile`` object in numpy).

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

        Attributes
        ----------
        files : list of str
            List of all files in the archive with a ``.npy`` extension.
        zip : ZipFile instance
            The ZipFile object initialized with the zipped archive.
        """
        _zip = make_zipfile(fid)

        self.files = _zip.namelist()
        self.zip = _zip

        if hasattr(fid, "close"):
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
        try:
            self.close()
        except AttributeError as e:
            if str(e) == "'ScpFile' object has no attribute 'zip'":
                pass
        except Exception as e:
            raise e

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):

        member = False
        ext = None

        if key in self.files:
            member = True
            _, ext = os.path.splitext(key)

        if member and ext in [".npy"]:
            f = self.zip.open(key)
            return read_array(f, allow_pickle=True)

        elif member and ext in [".scp"]:
            from spectrochempy.core.dataset.nddataset import NDDataset

            # f = io.BytesIO(self.zip.read(key))
            content = self.zip.read(key)
            return NDDataset.load(key, content=content)

        elif member and ext in [".json"]:
            content = self.zip.read(key)
            return json.loads(content, object_hook=json_decoder)

        elif member:
            return self.zip.read(key)

        else:
            raise KeyError("%s is not a file in the archive or is not " "allowed" % key)

    def __contains__(self, key):
        return self.files.__contains__(key)


# ======================================================================================================================
if __name__ == "__main__":
    pass

# EOF
