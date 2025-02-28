# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

r"""
NDDataset Input/Output Module.

This module implements input/output functionality for NDDataset objects. It provides:

- Save/load operations in native .scp format
- JSON serialization/deserialization
- Directory and file path management
- File type handling

Classes
-------
NDIO
    Base class providing I/O operations for NDDataset objects

Functions
---------
load(filename, **kwargs)
    Load a dataset from a .scp file
zipfile_factory(file, \*args, \*\*kwargs)
    Create a ZipFile with proper configuration
"""

__all__ = ["load"]
import io
import json
import os
import pathlib
import zipfile
from typing import Any
from typing import BinaryIO

import numpy as np
from traitlets import HasTraits
from traitlets import Instance
from traitlets import Unicode
from traitlets import Union

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.utils import exceptions
from spectrochempy.utils.file import check_filename_to_save
from spectrochempy.utils.file import pathclean
from spectrochempy.utils.jsonutils import json_encoder
from spectrochempy.utils.misc import TYPE_BOOL
from spectrochempy.utils.zip import ScpFile

# from warnings import warn

SCPY_SUFFIX: dict[str, str] = {"NDDataset": ".scp", "Project": ".pscp"}


def zipfile_factory(
    file: str | pathlib.Path | BinaryIO,
    *args: Any,
    **kwargs: Any,
) -> "zipfile.ZipFile":
    """
    Create a ZipFile with Zip64 support.

    Parameters
    ----------
    file : Union[str, pathlib.Path, BinaryIO]
        File path or file-like object
    *args : Any
        Additional arguments passed to ZipFile constructor
    **kwargs : Any
        Additional keyword arguments passed to ZipFile constructor

    Returns
    -------
    zipfile.ZipFile
        Configured zip file object

    """
    if not hasattr(file, "read"):
        file = os.fspath(file)
    import zipfile

    kwargs["allowZip64"] = True
    return zipfile.ZipFile(file, *args, **kwargs)


class NDIO(HasTraits):
    """
    Input/Output interface for NDDataset objects.

    This class provides methods for:
    - Saving/loading datasets in native format
    - Managing file paths and types
    - JSON serialization

    Properties
    ----------
    filename : pathlib.Path
        Current filename for this dataset
    directory : pathlib.Path
        Current directory for this dataset
    filetype : List[str]
        Supported file types
    suffix : str
        File extension
    """

    _filename = Union((Instance(pathlib.Path), Unicode()), allow_none=True)

    @property
    def directory(self):
        """
        Current directory for this dataset.

        ReadOnly property - automatically set when the filename is updated if
        it contains a parent on its path.
        """
        if self._filename:
            return pathclean(self._filename).parent
        return None

    @property
    def filename(self):
        """Current filename for this dataset."""
        if self._filename:
            try:
                return self._filename.relative_to(self.preferences.datadir)
            except ValueError:
                return self._filename
        else:
            return None

    @filename.setter
    def filename(self, val):
        self._filename = pathclean(val)

    @property
    def filetype(self):
        """Type of current file."""
        klass = self._implements()
        return [f"SpectroChemPy {klass} file (*{SCPY_SUFFIX[klass]})"]

    @property
    def suffix(self):
        """
        Filename suffix.

        Read Only property - automatically set when the filename is updated
        if it has a suffix, else give
        the default suffix for the given type of object.
        """
        if self._filename and self._filename.suffix:
            return self._filename.suffix
        klass = self._implements()
        return SCPY_SUFFIX[klass]

    # ----------------------------------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------------------------------
    def __dir__(self):
        return [
            "filename",
        ]

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def save(self, **kwargs: Any) -> pathlib.Path | None:
        """
        Save dataset in native .scp format.

        Parameters
        ----------
        **kwargs : Any
            Optional arguments passed to save_as()

        Returns
        -------
        Optional[pathlib.Path]
            Path to saved file if successful

        See Also
        --------
        save_as : Save with new filename
        write : Export to different format

        """
        # By default we save the file in the self.directory and with the
        # name + suffix depending
        # on the current object type
        if self.directory is None:
            filename = pathclean(".") / self.name
        else:
            filename = pathclean(self.directory) / self.name

        default_suffix = SCPY_SUFFIX[self._implements()]
        filename = filename.with_suffix(default_suffix)

        if not filename.exists() and kwargs.get("confirm", True):
            # never saved
            kwargs["caption"] = f"Save the current {self._implements()} as ... "
            return self.save_as(filename, **kwargs)

        # was already saved previously with this name,
        # in this case we do not display a dialog and overwrite the same file

        self.name = filename.stem
        return self.dump(filename, **kwargs)

    def save_as(self, filename="", **kwargs):
        """
        Save the current NDDataset in SpectroChemPy format (.scp).

        Parameters
        ----------
        filename : str
            The filename of the file where to save the current dataset.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Other Parameters
        ----------------
        directory : str, optional
            If specified, the given `directory` and the `filename` will be
            appended.

        See Also
        --------
        save : Save current dataset.
        write : Export current dataset to different format.

        Notes
        -----
        Adapted from :class:`numpy.savez` .

        Examples
        --------
        Read some data from an OMNIC file

        >>> nd = scp.read_omnic('wodger.spg')
        >>> assert nd.name == 'wodger'

        Write it in SpectroChemPy format (.scp)
        (return a `pathlib` object)

        >>> filename = nd.save_as('new_wodger')

        Check the existence of the scp file

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
        default_suffix = SCPY_SUFFIX[self._implements()]
        if filename is not None and not filename.is_dir():
            filename = filename.with_suffix(default_suffix)

        kwargs["filetypes"] = self.filetype
        kwargs["caption"] = f"Save the current {self._implements()} as ... "
        filename = check_filename_to_save(
            self,
            filename,
            save_as=True,
            suffix=default_suffix,
            **kwargs,
        )

        if filename:
            self.filename = filename
            return self.dump(filename, **kwargs)
        return None

    @classmethod
    def load(cls, filename, **kwargs):
        """
        Open data from a '*.scp' (NDDataset) or '*.pscp' (Project) file.

        Parameters
        ----------
        filename :  `str` , `pathlib` or `file` objects
            The name of the file to read (or a file objects).
        **kwargs
            Optional keyword parameters (see Other Parameters).

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
        Adapted from `numpy.load` .

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
        content = kwargs.get("content")

        if content:
            fid = io.BytesIO(content)
        else:
            # be sure to convert filename to a pathlib object with the
            # default suffix
            filename = pathclean(filename)
            suffix = cls().suffix
            filename = filename.with_suffix(suffix)
            if kwargs.get("directory") is not None:
                filename = pathclean(kwargs.get("directory")) / filename
            if not filename.exists():
                raise FileNotFoundError(f"No file with name {filename} could be found.")
                # filename = check_filenames(filename, **kwargs)[0]
            fid = open(filename, "rb")  # noqa: SIM115

        # get zip file
        try:
            obj = ScpFile(fid)
        except FileNotFoundError as e:
            raise exceptions.SpectroChemPyError(
                f"File {filename} doesn't exist!",
            ) from e
        except Exception as e:
            if str(e) == "File is not a zip file":
                raise exceptions.SpectroChemPyError(
                    "File not in 'scp' or 'pscp' format!",
                ) from e
            raise exceptions.SpectroChemPyError("Undefined error!") from e

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
        js = json_encoder(self, encoding=encoding)
        return json.dumps(js, indent=2)

    @classmethod
    def loads(cls, js: dict[str, Any]) -> Any:
        """
        Deserialize dataset from JSON.

        Parameters
        ----------
        js : Dict[str, Any]
            JSON object to deserialize

        Returns
        -------
        Any
            Deserialized dataset object

        Raises
        ------
        TypeError
            If JSON cannot be deserialized

        """
        from spectrochempy.core.dataset.nddataset import NDDataset
        from spectrochempy.core.project.project import Project
        from spectrochempy.core.script import Script

        # .........................
        def item_to_attr(obj, dic):
            for key, val in dic.items():
                try:
                    if "readonly" in dic and key in ["readonly", "name"]:
                        # case of the meta and preferences
                        pass

                    elif hasattr(obj, f"_{key}"):
                        # use the hidden attribute if it exists
                        key = f"_{key}"

                    if val is None:
                        pass

                    elif key in ["_meta", "_preferences"]:  # "_ranges",
                        setattr(obj, key, item_to_attr(getattr(obj, key), val))

                    elif key in ["_coordset"]:
                        _coords = []
                        for v in val["coords"]:
                            if "data" in v:
                                # coords
                                _coords.append(item_to_attr(Coord(), v))
                            elif "coords" in v:
                                # likely a coordset (multicoordinates)
                                if v["is_same_dim"]:
                                    _mcoords = []
                                    for mv in v["coords"]:
                                        _mcoords.append(item_to_attr(Coord(), mv))

                                    cs = CoordSet(*_mcoords[::-1], name=v["name"])
                                    _coords.append(cs)
                                else:
                                    raise ValueError("Invalid : not a multicoordinate")

                        coords = {c.name: c for c in _coords}
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

                    elif key in ["_history"]:
                        obj.history = val

                    else:
                        if isinstance(val, TYPE_BOOL) and key == "_mask":
                            val = np.bool_(val)
                        if isinstance(obj, NDDataset) and key == "_filename":
                            obj.filename = val  # This is a hack because for some reason fileame attribute is not
                            # found ????
                        else:
                            setattr(obj, key, val)

                except Exception as e:
                    raise TypeError(f"for {key} {e}") from e

            return obj

        # Create the class object and load it with the JSON content
        return item_to_attr(cls(), js)

    def dump(self, filename, **kwargs):
        """
        Save the current object into compressed native spectrochempy format.

        Parameters
        ----------
        filename: str of  `pathlib` object
            File name where to save the current object.

        """
        # Stage data in a temporary file on disk, before writing to zip.
        import tempfile
        import zipfile

        # prepare the json data
        # try:
        js = self.dumps(encoding="base64")
        # except Exception as e:
        #     warn(str(e))

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


load = NDIO.load  # make plot accessible directly from the scp API
