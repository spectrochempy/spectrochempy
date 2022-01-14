# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module define the class |NDIO| in which input/output standard
methods for a |NDDataset| are defined.
"""

__all__ = ["NDIO", "SCPY_SUFFIX"]

import io
import json
import pathlib

import numpy as np
from numpy.lib.npyio import zipfile_factory
from traitlets import HasTraits, Instance, Union, Unicode

from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.utils import (
    SpectroChemPyException,
    pathclean,
    ScpFile,
    check_filename_to_save,
    json_serialiser,
    TYPE_BOOL,
)

SCPY_SUFFIX = {"NDDataset": ".scp", "Project": ".pscp"}


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

# ======================================================================================================================
# Class NDIO to handle I/O of datasets
# ======================================================================================================================


class NDIO(HasTraits):
    """
    Import/export interface from |NDDataset|.

    This class is used as basic import/export interface of the |NDDataset|.
    """

    _filename = Union((Instance(pathlib.Path), Unicode()), allow_none=True)

    @property
    def directory(self):
        """
        `Pathlib` object - current directory for this dataset.

        ReadOnly property - automaticall set when the filename is updated if
        it contains a parent on its path.
        """
        if self._filename:
            return pathclean(self._filename).parent
        else:
            return None

    @property
    def filename(self):
        """
        `Pathlib` object - current filename for this dataset.
        """
        if self._filename:
            return self._filename.stem + self.suffix
        else:
            return None

    @filename.setter
    def filename(self, val):
        self._filename = pathclean(val)

    @property
    def filetype(self):
        klass = self.implements()
        return [f"SpectroChemPy {klass} file (*{SCPY_SUFFIX[klass]})"]

    @property
    def suffix(self):
        """
        filename suffix.

        Read Only property - automatically set when the filename is updated
        if it has a suffix, else give
        the default suffix for the given type of object.
        """
        if self._filename and self._filename.suffix:
            return self._filename.suffix
        else:
            klass = self.implements()
            return SCPY_SUFFIX[klass]

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __dir__(self):
        return [
            "filename",
        ]

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    def save(self, **kwargs):
        """
        Save the current object in SpectroChemPy format.

        Default extension is *.scp for |NDDataset|'s and *.pscp for
        |Project|'s.

        Parameters
        ----------
        **kwargs : dict
          See other parameters.

        Other Parameters
        ----------------
        confirm : bool

        See Also
        ---------
        save_as : Save current object with a different name and/or directory.
        write : Export current object to different format.

        Examples
        ---------

        Read some data from an OMNIC file
        >>> nd = scp.read_omnic('wodger.spg')
        >>> assert nd.name == 'wodger'

        Write it in SpectroChemPy format (.scp)
        (return a `pathlib` object)
        >>> filename = nd.save()

        Check the existence of the scp fie
        >>> assert filename.is_file()
        >>> assert filename.name == 'wodger.scp'

        Remove this file
        >>> filename.unlink()
        """

        # By default we save the file in the self.directory and with the
        # name + suffix depending
        # on the current object type
        if self.directory is None:
            filename = pathclean(".") / self.name
        else:
            filename = pathclean(self.directory) / self.name

        default_suffix = SCPY_SUFFIX[self.implements()]
        filename = filename.with_suffix(default_suffix)

        if not filename.exists() and kwargs.get("confirm", True):
            # never saved
            kwargs["caption"] = f"Save the current {self.implements()} as ... "
            return self.save_as(filename, **kwargs)

        # was already saved previously with this name,
        # in this case we do not display a dialog and overwrite the same file

        self.name = filename.stem
        return self.dump(filename, **kwargs)

    # ..........................................................................
    def save_as(self, filename="", **kwargs):
        """
        Save the current |NDDataset| in SpectroChemPy format (*.scp).

        Parameters
        ----------
        filename : str
            The filename of the file where to save the current dataset.
        **kwargs : dict
            See Other Parameters.

        Other Parameters
        -----------------
        directory : str, optional
            If specified, the given `directory` and the `filename` will be
            appended.

        See Also
        ---------
        save : Save current dataset.
        write : Export current dataset to different format.

        Notes
        -----
        Adapted from :class:`numpy.savez`.

        Examples
        --------

        Read some data from an OMNIC file
        >>> nd = scp.read_omnic('wodger.spg')
        >>> assert nd.name == 'wodger'

        Write it in SpectroChemPy format (.scp)
        (return a `pathlib` object)
        >>> filename = nd.save_as('new_wodger')

        Check the existence of the scp fie
        >>> assert filename.is_file()
        >>> assert filename.name == 'new_wodger.scp'

        Remove this file
        >>> filename.unlink()
        """
        if filename:
            # we have a filename
            # by default it use the saved directory
            filename = pathclean(filename)
            if self.directory and self.directory != filename.parent:
                filename = self.directory / filename
        else:
            filename = self.directory

        # suffix must be specified which correspond to the type of the
        # object to save
        default_suffix = SCPY_SUFFIX[self.implements()]
        if filename is not None and not filename.is_dir():
            filename = filename.with_suffix(default_suffix)

        kwargs["filetypes"] = self.filetype
        kwargs["caption"] = f"Save the current {self.implements()} as ... "
        filename = check_filename_to_save(
            self, filename, save_as=True, suffix=default_suffix, **kwargs
        )

        if filename:
            self.filename = filename
            return self.dump(filename, **kwargs)

    # ..........................................................................
    @classmethod
    def load(cls, filename, **kwargs):
        """
        Open data from a '*.scp' (NDDataset) or '.pscp' (Project) file.

        Parameters
        ----------
        filename :  `str`, `pathlib` or `file` objects
            The name of the file to read (or a file objects.
        **kwargs : dict, optional
            Any additional keyword(s) to pass to the actual reader.
            See Other Parameters.

        Other Parameters
        ----------------
        content : str, optional
             The optional content of the file(s) to be loaded as a binary string.

        See Also
        --------
        read : Import dataset from various orgines.
        save : Save the current dataset.

        Notes
        -----
        Adapted from `numpy.load`.

        Examples
        --------

        >>> nd1 = scp.read('irdata/nh4y-activation.spg')
        >>> f = nd1.save()
        >>> f.name
        'nh4y-activation.scp'
        >>> nd2 = scp.load(f)

        Alternatively, this method can be called as a class method of NDDataset or Project object:

        >>> from spectrochempy import *
        >>> nd2 = NDDataset.load(f)
        """
        content = kwargs.get("content", None)

        if content:
            fid = io.BytesIO(content)
        else:
            # be sure to convert filename to a pathlib object with the
            # default suffix
            filename = pathclean(filename)
            suffix = cls().suffix
            filename = filename.with_suffix(suffix)
            if kwargs.get("directory", None) is not None:
                filename = pathclean(kwargs.get("directory")) / filename
            if not filename.exists():
                raise FileNotFoundError(f"No file with name {filename} could be found.")
                # filename = check_filenames(filename, **kwargs)[0]
            fid = open(filename, "rb")

        # get zip file
        try:
            obj = ScpFile(fid)
        except FileNotFoundError:
            raise SpectroChemPyException(f"File {filename} doesn't exist!")
        except Exception as e:
            if str(e) == "File is not a zip file":
                raise SpectroChemPyException("File not in 'scp' or 'pscp' format!")
            raise SpectroChemPyException("Undefined error!")

        js = obj[obj.files[0]]
        if kwargs.get("json", False):
            return js

        new = cls.loads(js)

        fid.close()

        if filename:
            filename = pathclean(filename)
            new._filename = filename
            new.name = filename.stem

        return new

    def dumps(self, encoding=None):

        js = json_serialiser(self, encoding=encoding)
        return json.dumps(js, indent=2)

    @classmethod
    def loads(cls, js):

        from spectrochempy.core.project.project import Project
        from spectrochempy.core.dataset.nddataset import NDDataset
        from spectrochempy.core.scripts.script import Script

        # .........................
        def item_to_attr(obj, dic):

            for key, val in dic.items():

                try:
                    if "readonly" in dic.keys() and key in ["readonly", "name"]:
                        # case of the meta and preferences
                        pass

                    elif hasattr(obj, f"_{key}"):
                        # use the hidden attribute if it exists
                        key = f"_{key}"

                    if val is None:
                        pass

                    elif key in ["_meta", "_ranges", "_preferences"]:
                        setattr(obj, key, item_to_attr(getattr(obj, key), val))

                    elif key in ["_coordset"]:
                        _coords = []
                        for v in val["coords"]:
                            if "data" in v:
                                _coords.append(item_to_attr(Coord(), v))
                            else:
                                _coords.append(item_to_attr(LinearCoord(), v))

                        if val["is_same_dim"]:
                            obj.set_coordset(_coords)
                        else:
                            coords = dict((c.name, c) for c in _coords)
                            obj.set_coordset(coords)
                        obj._name = val["name"]
                        obj._references = val["references"]

                    elif key in ["_datasets"]:
                        # datasets = [item_to_attr(NDDataset(name=k),
                        # v) for k, v in val.items()]
                        datasets = [item_to_attr(NDDataset(), js) for js in val]
                        obj.datasets = datasets

                    elif key in ["_projects"]:
                        projects = [item_to_attr(Project(), js) for js in val]
                        obj.projects = projects

                    elif key in ["_scripts"]:
                        scripts = [item_to_attr(Script(), js) for js in val]
                        obj.scripts = scripts

                    elif key in ["_parent"]:
                        # automatically set
                        pass

                    else:
                        if isinstance(val, TYPE_BOOL) and key == "_mask":
                            val = np.bool_(val)
                        if isinstance(obj, NDDataset) and key == "_filename":
                            obj.filename = val  # This is a hack because for some reason fileame attribute is not
                            # found ????
                        else:
                            setattr(obj, key, val)

                except Exception as e:
                    raise TypeError(f"for {key} {e}")

            return obj

        # Create the class object and load it with the JSON content
        new = item_to_attr(cls(), js)

        return new

    # ..........................................................................
    def dump(self, filename, **kwargs):
        """
        Save the current object into compressed native spectrochempy format.

        Parameters
        ----------
        filename: str of  `pathlib` object
            File name where to save the current object.
        """

        # Stage data in a temporary file on disk, before writing to zip.
        import zipfile
        import tempfile

        # prepare the json data
        try:
            js = self.dumps(encoding="base64")
        except Exception as e:
            print(e)

        # write in a temp file
        _, tmpfile = tempfile.mkstemp(suffix="-spectrochempy")
        tmpfile = pathclean(tmpfile)
        tmpfile.write_bytes(js.encode("utf-8"))

        # compress and write zip file
        zipf = zipfile_factory(filename, mode="w", compression=zipfile.ZIP_DEFLATED)
        zipf.write(tmpfile, arcname=f"{self.name}.json")
        # tmpfile.unlink()
        zipf.close()

        self.filename = filename
        self.name = filename.stem

        return filename


# ======================================================================================================================
if __name__ == "__main__":
    pass

# EOF
