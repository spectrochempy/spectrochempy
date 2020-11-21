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
# __dataset_methods__ = ['write', 'load']

from os import close as osclose, remove as osremove
import io
import datetime
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.compat import asstr
from numpy.lib.format import write_array
from numpy.lib.npyio import zipfile_factory, NpzFile
from traitlets import HasTraits, Unicode

from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.utils import SpectroChemPyException
from spectrochempy.utils import pathclean, Meta, readdirname, check_filenames, check_filename_to_open
from spectrochempy.units import Unit, Quantity
from spectrochempy.core.readers.importer import _Importer

# ==============================================================================
# Class NDIO to handle I/O of datasets
# ==============================================================================

class NDIO(HasTraits):
    """
    Import/export interface from |NDDataset|

    This class is used as basic import/export interface of the |NDDataset|.

    """

    _filename = Unicode()

    @property
    def filename(self):
        """
        str - current filename for this dataset.

        """
        if self._filename:
            return pathclean(self._filename).name
        else:
            return None

    @filename.setter
    def filename(self, fname):
        self._filename = fname

    @property
    def directory(self):
        """
        `Pathlib` object - current directory for this dataset.

        """
        if self._filename:
            return pathclean(self._filename).parent
        else:
            return None

    # ------------------------------------------------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------------------------------------------------

    def __dir__(self):
        return ['filename', ]

    # ------------------------------------------------------------------------------------------------------------------
    # Generic save function
    # ------------------------------------------------------------------------------------------------------------------

    def save(self, filename=None, **kwargs):
        """
        Save the current |NDDataset| (default extension : ``.scp`` ).

        Parameters
        ----------
        filename : str
            The filename of the file where to save the current dataset
        directory : str, optional
            If specified, the given `directory` and the `filename` will be
            appended.

        Examples
        ---------
        Read some experimental data and then save in our proprietary format
        **scp**

        >>> from spectrochempy import NDDataset #doctest: +ELLIPSIS

        >>> mydataset = NDDataset.read_omnic('irdata/nh4y-activation.spg')
        >>> mydataset.save('mydataset.scp')

        Notes
        -----
        adapted from :class:`numpy.savez`

        See Also
        ---------
        write

        """

        filename = pathclean(filename)
        same_dir = kwargs.pop('same_dir', False)
        directory = kwargs.get('directory',
                               self.directory if same_dir else Path.cwd())

        if not filename:
            # the current file name or default filename (id)
            filename = pathclean(self.filename)
            if self.directory:
                directory = pathclean(self.directory)

        if not filename.suffix == '.scp':
            filename = filename.with_suffix('.scp')

        if directory:
            filename = directory / filename

        directory = readdirname(filename.parent)
        if directory is not None:
            filename = directory / filename.name

        # Stage data in a temporary file on disk, before writing to zip.
        import zipfile
        import tempfile
        zipf = zipfile_factory(filename, mode="w",
                               compression=zipfile.ZIP_DEFLATED)
        fd, tmpfile = tempfile.mkstemp(suffix='-spectrochempy.scp')
        osclose(fd)

        pars = {}
        objnames = self.__dir__()

        def _loop_on_obj(_names, obj, level=''):
            # Recursive scan on NDDataset objects

            for key in _names:

                val = getattr(obj, f"_{key}")

                if isinstance(val, np.ndarray):

                    with open(tmpfile, 'wb') as fid:
                        write_array(fid, np.asanyarray(val), allow_pickle=True)

                    zipf.write(tmpfile, arcname=level + key + '.npy')

                elif isinstance(val, CoordSet):

                    for v in val._coords:
                        _objnames = dir(v)
                        if isinstance(v, Coord):
                            _loop_on_obj(_objnames, obj=v,
                                         level=f"coord_{v.name}_")
                        elif isinstance(v, CoordSet):
                            _objnames.remove('coords')
                            _loop_on_obj(_objnames, obj=v,
                                         level=f"coordset_{v.name}_")
                            for vi in v:
                                _objnames = dir(vi)
                                _loop_on_obj(_objnames, obj=vi,
                                             level=f"coordset_{v.name}_"
                                                   f"coord_{vi.name[1:]}_")

                elif isinstance(val, datetime.datetime):

                    pars[level + key] = val.timestamp()

                elif isinstance(val, np.dtype):

                    pars[level + key] = str(val)

                elif isinstance(val, Unit):

                    pars[level + key] = str(val)

                elif isinstance(val, Meta):
                    d = val.to_dict()
                    # we must handle Quantities
                    for k, v in d.items():
                        if isinstance(v, list):
                            for i, item in enumerate(v):
                                if isinstance(item, Quantity):
                                    item = list(item.to_tuple())
                                    if isinstance(item[0], np.ndarray):
                                        item[0] = item[0].tolist()
                                    d[k][i] = tuple(item)
                    pars[level + key] = d

                elif val is None:
                    continue

                elif isinstance(val, dict) and key == 'axes':
                    # do not save the matplotlib axes
                    continue

                elif isinstance(val, (plt.Figure, plt.Axes)):
                    # pass the figures and Axe
                    continue

                else:
                    pars[level + key] = val

        _loop_on_obj(objnames, self)

        with open(tmpfile, 'w') as f:
            f.write(json.dumps(pars))

        zipf.write(tmpfile, arcname='pars.json')

        osremove(tmpfile)

        zipf.close()

        return filename

    # ------------------------------------------------------------------------------------------------------------------
    # Generic load function
    # ------------------------------------------------------------------------------------------------------------------

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
        >>> from spectrochempy import *
        >>> mydataset = NDDataset.load('mydataset.scp')
        >>> print(mydataset)
        <BLANKLINE>
        ...

        Notes
        -----
        adapted from `numpy.load`

        See Also
        --------
        read, save


        """
        content = kwargs.get('content', None)

        if content:
            fid = io.BytesIO(content)
        else:
            # be sure to convert filename to a pathlib object
            filename = pathclean(filename)
            if not filename.exists():
                filename = check_filenames(filename.with_suffix('.scp'), **kwargs)[0]
            fid = open(filename, 'rb')

        # get zip file
        try:
            obj = NpzFile(fid, allow_pickle=True)
        except FileNotFoundError:
            raise SpectroChemPyException(f"File {filename} doesn't exist!")
        except Exception as e:
            if str(e) == 'File is not a zipfile':
                raise SpectroChemPyException("File not in 'scp' format!")
            raise SpectroChemPyException("Undefined error!")

        # interpret
        coords = None
        new = cls()

        for key, val in list(obj.items()):
            if key.startswith('coord_'):
                if not coords:
                    coords = {}
                els = key.split('_')
                dim = els[1]
                if dim not in coords.keys():
                    coords[dim] = Coord()
                setattr(coords[dim], "_%s" % els[2], val)

            if key.startswith('coordset_'):
                if not coords:
                    coords = {}
                els = key.split('_')
                dim = els[1]
                idx = "_" + els[3]
                if dim not in coords.keys():
                    coords[dim] = CoordSet({
                            idx: Coord()
                            })
                if idx not in coords[dim].names:
                    coords[dim].set(**{
                            idx: Coord()
                            })
                setattr(coords[dim][idx], "_%s" % els[4], val)

            elif key == "pars.json":
                pars = json.loads(asstr(val))
            else:
                setattr(new, "_%s" % key, val)

        fid.close()

        def setattributes(clss, key, val):
            # utility function to set the attributes
            if key in ['modified', 'date']:
                val = datetime.datetime.fromtimestamp(val)
                setattr(clss, "_%s" % key, val)
            elif key == 'meta':
                # handle the case were quantity were saved
                for k, v in val.items():
                    if isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, (list, tuple)):
                                try:
                                    v[i] = Quantity.from_tuple(item)
                                except TypeError:
                                    # not a quantity
                                    pass
                        val[k] = v
                clss.meta.update(val)
            elif key == 'plotmeta':
                # handle the case were quantity were saved
                for k, v in val.items():
                    if isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, (list, tuple)):
                                try:
                                    v[i] = Quantity.from_tuple(item)
                                except TypeError:
                                    # not a quantity
                                    pass
                        val[k] = v
                clss.plotmeta.update(val)
            elif key in ['units']:
                setattr(clss, key, val)
            elif key in ['dtype']:
                setattr(clss, "_%s" % key, np.dtype(val))
            else:
                setattr(clss, "_%s" % key, val)

        for key, val in list(pars.items()):

            if key.startswith('coord_'):

                els = key.split('_')
                dim = els[1]
                setattributes(coords[dim], els[2], val)

            elif key.startswith('coordset_'):
                els = key.split('_')
                dim = els[1]
                if key.endswith("is_same_dim"):
                    setattributes(coords[dim], "is_same_dim", val)
                elif key.endswith("name"):
                    setattributes(coords[dim], "name", val)
                elif key.endswith("references"):
                    setattributes(coords[dim], "references", val)
                else:
                    idx = "_" + els[3]
                    setattributes(coords[dim][idx], els[4], val)
            else:

                setattributes(new, key, val)

        if filename:
            new._filename = str(filename)

        if coords:
            new.set_coords(coords)

        return new


    # ------------------------------------------------------------------------------------------------------------------
    # Generic write function
    # ------------------------------------------------------------------------------------------------------------------

    def write(self, *args, **kwargs):
        """
        Generic write function which actually delegate the work to an
        writer defined by the parameter ``protocol``.

        Parameters
        ----------
        filename : str
            The path to the file to be read
        protocol : str
            The protocol used to write the
            |NDDataset| in a file,
            which will determine the exporter to use.
        kwargs : optional keyword parameters
            Any additional keyword to pass to the actual exporter

        See Also
        --------
        save

        """

        filename = None
        if args:
            filename = args[0]
        filename = kwargs.pop('filename', filename)

        # kwargs only
        protocol = kwargs.pop('protocol', None)
        if kwargs.get('to_string', False):
            # json by default if we output a string
            protocol = 'json'

        if not protocol:
            # try to estimate the protocol from the file name extension
            extension = filename.suffix
            if len(extension) > 0:
                protocol = extension[1:].lower()

        if protocol == 'scp':
            return self.save(filename)

        # find the adequate reader
        try:
            # find the adequate reader
            _writer = getattr(self, 'write_{}'.format(protocol))
            return _writer(filename=filename, **kwargs)

        except Exception as e:
            raise AttributeError(f'The specified writter for protocol `{protocol}` was not found!')



# ==============================================================================

if __name__ == '__main__':
    pass
