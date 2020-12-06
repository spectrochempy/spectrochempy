# -*- coding: utf-8 -*-
# ==============================================================================
#  Copyright (©) 2015-2020
#  LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory
# ==============================================================================

"""
This module define the class |NDIO| in which input/output standard
methods for a |NDDataset| are defined.

"""

__all__ = ['NDIO']

import io
import datetime
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.format import write_array
from numpy.lib.npyio import zipfile_factory
from traitlets import HasTraits, Instance, Unicode

from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.utils import SpectroChemPyException
from spectrochempy.units import Quantity
from spectrochempy.utils import pathclean, Meta, check_filenames, ScpFile, check_filename_to_save
from spectrochempy.utils import json_serialiser, json_decoder


# ==============================================================================
# Class NDIO to handle I/O of datasets
# ==============================================================================

class NDIO(HasTraits):
    """
    Import/export interface from |NDDataset|

    This class is used as basic import/export interface of the |NDDataset|.

    """

    _filename = Instance(pathlib.Path, allow_none=True)

    @property
    def directory(self):
        """
        `Pathlib` object - current directory for this dataset

        ReadOnly property - automaticall set when the filename is updated if it contains a parent on its path

        """
        if self._filename:
            return pathclean(self._filename).parent.resolve()
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
        if self.implements('Project'):
            return ['SpectroChemPy Project file (*.pscp)']
        elif self.implements('NDPanel'):
            return ['SpectroChemPy Panel file (*.nscp)']
        elif self.implements('NDDataset'):
            return ['SpectroChemPy dataset file (*.scp)']


    @property
    def suffix(self):
        """
        filename suffix

        Read Only property - automatically set when the filename is updated if it has a suffix, else give
        the default suffix for the given type of object.

        """
        if self._filename and  self._filename.suffix:
            return self._filename.suffix
        else:
            if self.implements('Project'):
                suffix = ".pscp"
            elif self.implements('NDPanel'):
                suffix = ".nscp"
            elif self.implements('NDDataset'):
                suffix = ".scp"
            return suffix

    # ------------------------------------------------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------------------------------------------------

    def __dir__(self):
        return ['filename', ]

    # ------------------------------------------------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def save(self, **kwargs):
        """
        Save the current object in SpectroChemPy format.

        Default extension is *.scp for |NDDataset|'s, *.nscp for |NDPanel|'s, and *.pscp for |Project|'s.

        See Also
        ---------
        save_as : save current object with a different name and/or directory
        write : export current object to different format

        Examples
        ---------

        read some data from an OMNIC file
        >>> import spectrochempy as scp
        >>> nd = scp.read_omnic('wodger.spg')
        >>> assert nd.name == 'wodger'

        write it in SpectroChemPy format (.scp)
        (return a `pathlib` object)
        >>> filename = nd.save()

        check the existence of the scp fie
        >>> assert filename.is_file()
        >>> assert filename.name == 'wodger.scp'

        Remove this file
        >>> filename.unlink()

        """

        # by default we save the file in the self.directory and with the name + suffix depending
        # on the current object type
        if self.directory is None:
            self.filename = pathclean('.') / self.name

        filename = self.directory / self.filename

        if not filename.exists():
            # never saved
            kwargs['caption'] = f'Save the current {self.implements()} as ... '
            return self.save_as(filename, **kwargs)

        # was already saved previously with this name,
        # in this case we do not display a dialog and overwrite the same file
        return self._save(filename, **kwargs)

    # ..................................................................................................................
    def save_as(self, filename='', **kwargs):
        """
        Save the current |NDDataset| in SpectroChemPy format (*.scp)

        Parameters
        ----------
        filename : str
            The filename of the file where to save the current dataset
        directory : str, optional
            If specified, the given `directory` and the `filename` will be
            appended.

        Examples
        ---------

        read some data from an OMNIC file
        >>> import spectrochempy as scp
        >>> nd = scp.read_omnic('wodger.spg')
        >>> assert nd.name == 'wodger'

        write it in SpectroChemPy format (.scp)
        (return a `pathlib` object)
        >>> filename = nd.save_as('new_wodger')

        check the existence of the scp fie
        >>> assert filename.is_file()
        >>> assert filename.name == 'new_wodger.scp'

        Remove this file
        >>> filename.unlink()

        Notes
        -----
        adapted from :class:`numpy.savez`

        See Also
        ---------
        save : save current dataset
        write : export current dataset to different format

        """
        if filename:
            # we have a filename
            # by default it use the saved directory
            filename = pathclean(filename)
            if self.directory and self.directory != filename.parent.resolve():
                filename = self.directory / filename
        else:
            filename = self.directory

        kwargs['filetypes'] = self.filetype
        kwargs['caption'] = f'Save the current {self.implements()} as ... '
        filename = check_filename_to_save(self, filename, save_as=True, **kwargs)

        if filename:
            self.filename = filename
            return self._save(filename, **kwargs)

    # ..................................................................................................................
    @classmethod
    def load(cls, filename, **kwargs):
        """
        Load a data from a '*.scp' file.

        It's a class method, that can be used directly on the class,
        without prior opening of a class instance.

        Parameters
        ----------
        filename :  `str`, `pathlib` or `file` objects
            The name of the file to read (or a file objects.
        content : str, optional
             The optional content of the file(s) to be loaded as a binary string
        kwargs : optional keyword parameters.
            Any additional keyword(s) to pass to the actual reader.


        Examples
        --------
        >>> from spectrochempy import NDDataset
        >>> nd = NDDataset.load('irdata/nh4y.scp')
        >>> print(nd)
        NDDataset: [float32] a.u. (shape: (y:55, x:5549))


        Notes
        -----
        adapted from `numpy.load`

        See Also
        --------
        read : import dataset from various orgines
        save : save the current dataset


        """
        content = kwargs.get('content', None)


        if content:
            fid = io.BytesIO(content)
        else:
            # be sure to convert filename to a pathlib object with the default suffix
            filename = pathclean(filename)
            suffix = cls().suffix
            if not filename.suffix:
                filename.suffix = suffix
            else:
                filename = filename.with_suffix(suffix)
            if kwargs.get('directory', None) is not None:
                filename = pathclean(kwargs.get('directory')) / filename
            if not filename.exists():
                filename = check_filenames(filename, **kwargs)[0]
            fid = open(filename, 'rb')

        # get zip file
        try:
            obj = ScpFile(fid)
        except FileNotFoundError:
            raise SpectroChemPyException(f"File {filename} doesn't exist!")
        except Exception as e:
            if str(e) == 'File is not a zip file':
                raise SpectroChemPyException("File not in 'scp' format!")
            raise SpectroChemPyException("Undefined error!")

        js = obj[obj.files[0]]
        new = cls.from_json(js)

        fid.close()

        if filename:
            filename = pathclean(filename)
            new._filename = filename
            new.name = filename.stem
        return new

    def to_json(self):

        objnames = dir(self)

        def obj_to_json(obj):

            objnames = dir(obj)


        def _loop_on_obj(_names, obj=self, parent={}):

            parent['type'] = self.__class__.__name__

            for key in _names:

                val = getattr(obj, "_%s" % key)
                if val is None:
                    # ignore None - when reading if something is missing it
                    # will be considered as None anyways
                    continue

                elif key == 'projects':
                    parent[key] = {}
                    for k, proj in val.items():
                        _objnames = dir(proj)
                        _loop_on_obj(_objnames, obj=proj, parent=parent[key])
                        parent[key][k] = projj

                elif key == 'datasets':
                    parent[key] = []
                    for k, ds in val.items():
                        dsj = ds.to_json()
                        parent[key].append( {k:dsj} )

                elif key == 'scripts':
                    parent[key] = {}
                    for k, sc in val.items():
                        _objnames = dir(sc)
                        scj = _loop_on_obj(_objnames, obj=sc, parent=parent[key])
                        parent[key][k] = scj

                elif isinstance(val, Meta):
                    parent[key] = val.to_dict()

                elif key == 'parent':
                    parent[key] = main

                else:
                    # probably some string
                    parent[key] = val

        # Recursive scan on Project content
        main = _loop_on_obj(objnames)

        return main





# ----------------------------------------------------------------------------------------------------------------------
# Private
# ----------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def _save(self, filename, **kwargs):
        # machinery to save the current dataset into native spectrochempy format

        # Stage data in a temporary file on disk, before writing to zip.
        import zipfile
        import tempfile

        zipf = zipfile_factory(filename, mode="w", compression=zipfile.ZIP_DEFLATED)
        _, tmpfile = tempfile.mkstemp(suffix='-spectrochempy')

        tmpfile = pathclean(tmpfile)

        js = json.dumps(self.to_json(), default=json_serialiser, indent=2)

        tmpfile.write_bytes(js.encode('utf-8'))

        zipf.write(tmpfile, arcname=f'{self.name}.json')

        tmpfile.unlink()

        zipf.close()

        self.filename = filename
        self.name = filename.stem

        return filename


# ======================================================================================================================
if __name__ == '__main__':
    pass

# EOF
