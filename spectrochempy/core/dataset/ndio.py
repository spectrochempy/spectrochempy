# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module define the class |NDIO| in which input/output standard
methods for a |NDDataset| are defined.

"""

__all__ = ['NDIO']

__dataset_methods__ = []

# ----------------------------------------------------------------------------------------------------------------------
# Python imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import datetime
import json
import warnings

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from numpy.compat import asbytes, asstr
from numpy.lib.format import write_array, MAGIC_PREFIX
from numpy.lib.npyio import zipfile_factory, NpzFile
from traitlets import HasTraits, Unicode

# ----------------------------------------------------------------------------------------------------------------------
# local import
# ----------------------------------------------------------------------------------------------------------------------

from .ndarray import NDArray
from .ndcoords import Coord, CoordSet
from spectrochempy.utils import SpectroChemPyWarning
from spectrochempy.utils.meta import Meta
from spectrochempy.units import Unit, Quantity
from spectrochempy.core import log, general_preferences as prefs


# ==============================================================================
# Class NDIO to handle I/O of datasets
# ==============================================================================

class NDIO(HasTraits):
    """
    Import/export interface from |NDDataset|

    This class is used as basic import/export interface of the |NDDataset|.

    """

    _filename = Unicode

    @property
    def filename(self):
        """
        str - current filename for this dataset.

        """
        if self._filename:
            return os.path.basename(self._filename)
        else:
            return self.name

    @filename.setter
    def filename(self, fname):
        self._filename = fname

    @property
    def directory(self):
        """
        str - current directory for this dataset.

        """
        if self._filename:
            return os.path.dirname(self._filename)
        else:
            return ''

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------

    def __dir__(self):
        return ['filename', ]

    # ------------------------------------------------------------------------------------------------------------------
    # Generic save function
    # ------------------------------------------------------------------------------------------------------------------

    def save(self, filename='', directory=prefs.datadir,
             **kwargs
             ):
        """
        Save the current |NDDataset| (default extension: ``.scp`` ).

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

        >>> from spectrochempy import * #doctest: +ELLIPSIS

        >>> mydataset = NDDataset.read_omnic('irdata/nh4y-activation.spg')
        >>> mydataset.save('mydataset.scp')

        Notes
        -----
        adapted from :class:`numpy.savez`

        See Also
        ---------
        write

        """

        directory = kwargs.get("directory", prefs.datadir)

        if not filename:
            # the current file name or default filename (id)
            filename = self.filename
            if self.directory:
                directory = self.directory

        if not os.path.exists(directory):
            raise IOError("directory doesn't exists!")

        if not filename.endswith('.scp'):
            filename = filename + '.scp'

        if os.path.isdir(directory):
            filename = os.path.expanduser(os.path.join(directory, filename))
        else:
            warnings.warn('Provided directory is a file, '
                          'so we use its parent directory',
                          SpectroChemPyWarning)
            filename = os.path.join(os.path.dirname(directory), filename)

        # Import is postponed to here since zipfile depends on gzip, an optional
        # component of the so-called standard library.
        import zipfile
        # Import deferred for startup time improvement
        import tempfile

        zipf = zipfile_factory(filename, mode="w",
                               compression=zipfile.ZIP_DEFLATED)

        # Stage data in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-spectrochempy.scp')
        os.close(fd)

        pars = {}
        objnames = self.__dir__()

        def _loop_on_obj(_names, obj=self, level=''):
            """Recursive scan on NDDataset objects"""

            for key in _names:

                val = getattr(obj, "_%s" % key)

                if isinstance(val, np.ndarray):

                    with open(tmpfile, 'wb') as fid:
                        write_array(fid, np.asanyarray(val), allow_pickle=True)

                    zipf.write(tmpfile, arcname=level + key + '.npy')

                elif isinstance(val, Coord):

                    _objnames = dir(val)
                    _loop_on_obj(_objnames, level=key + '.')

                elif isinstance(val, CoordSet):

                    for i, val in enumerate(val._coords):
                        _objnames = dir(val)
                        _loop_on_obj(_objnames, obj=val, level="coord_%d_" % i)

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

        _loop_on_obj(objnames)

        with open(tmpfile, 'w') as f:
            f.write(json.dumps(pars))

        zipf.write(tmpfile, arcname='pars.json')

        os.remove(tmpfile)

        zipf.close()

        self._filename = filename

    # ------------------------------------------------------------------------------------------------------------------
    # Generic load function
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _load(cls,
              fid='',
              protocol=None,
              directory=prefs.datadir,
              **kwargs
              ):

        filename = None

        # case where load was call directly from the API
        # e.g.,  A= scp.load("dataset.scp")
        # In this cas ewe need to define cls as a NDDataset class
        if isinstance(cls(), NDIO):
            # not run as a class method of NDDataset
            from spectrochempy import NDDataset
            cls = NDDataset

        if protocol is not None and protocol not in ['scp']:
            # TODO : case where fp is a file object
            filename = fid
            return cls.read(filename, protocol=protocol)

        if isinstance(fid, str) and protocol is None:
            filename, ext = os.path.splitext(fid)
            try:
                return cls.read(fid, protocol=ext[1:])
            except:
                pass

        if isinstance(fid, str):
            # this is a filename

            filename = fid
            directory = kwargs.get("directory", prefs.datadir)
            if not filename:
                raise IOError('no filename provided!')
            else:
                filename = os.path.expanduser(
                    os.path.join(directory, filename))
                try:
                    # cast to file in the testdata directory
                    # TODO: add possibility to search in several directory
                    fid = open(filename, 'rb')
                except:
                    if not filename.endswith('.scp'):
                        filename = filename + '.scp'
                    try:
                        # try again
                        fid = open(filename, 'rb')
                    except IOError:
                        raise IOError('no valid filename provided')

        # get zip file
        obj = NpzFile(fid, allow_pickle=True)

        log.debug(str(obj.files) + '\n')

        # interpret
        ndim = obj["data"].ndim
        coords = None
        new = cls()

        torem = []
        for key, val in list(obj.items()):
            if key.startswith('coord_'):
                if not coords:
                    coords = [Coord() for _ in range(ndim)]
                els = key.split('_')
                idx = int(els[1])
                # if obj["data"].shape[idx]==1:
                #    torem.append(idx)
                #    idx+=1
                setattr(coords[idx], "_%s" % els[2], val)

            elif key == "pars.json":
                pars = json.loads(asstr(val))
            else:
                setattr(new, "_%s" % key, val)

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
                idx = int(els[1])
                if obj["data"].shape[idx] == 1:
                    idx += 1
                setattributes(coords[idx], els[2], val)

            else:

                setattributes(new, key, val)

        if filename:
            new._filename = filename

        if coords:
            ncoords = []
            for idx, v in enumerate(coords):
                if idx not in torem:
                    ncoords.append(coords[idx])
            new.coords = ncoords

        return new

    @classmethod
    def load(cls,
             fid='',
             protocol=None,
             directory=prefs.datadir,
             **kwargs
             ):
        """
        Load a list of dataset objects saved as a pickle files ( e.g., '\*.scp' file).

        It's a class method, that can be used directly on the class,
        without prior opening of a class instance.

        Parameters
        ----------
        fid : list of `str` or `file` objects
            The names of the files to read (or the file objects).
        protocol : str, optional, default:'scp'
            The default type for saving.
        directory : str, optional, default:`prefs.datadir`
            The directory from where to load the file.
        kwargs : optional keyword parameters.
            Any additional keyword(s) to pass to the actual reader.


        Examples
        --------
        >>> from spectrochempy import *
        >>> mydataset = NDDataset.load('mydataset.scp')
        >>> print(mydataset)
        <BLANKLINE>
        ...

        by default, directory for saving is the `data`.
        So the same thing can be done simply by:

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
        datasets = []
        files = fid
        if not isinstance(fid, (list, tuple)):
            files = [fid]

        for f in files:
            nd = NDIO._load(cls, fid=f, protocol=protocol, directory=directory, **kwargs)
            datasets.append(nd)

            # TODO: allow a concatenation or stack if possible

        if len(datasets) == 1:
            return datasets[0]

    # ------------------------------------------------------------------------------------------------------------------
    # Generic read function
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def read(cls, filename=None, **kwargs):
        """
        Generic read function. It's like `load` a class method.

        Parameters
        ----------
        filename : str
            The path to the file to be read
        protocol : str
            Protocol used for reading. If not provided, the correct protocol
            is evaluated from the file name extension.
        kwargs : optional keyword parameters
            Any additional keyword to pass to the actual reader

        See Also
        --------
        load

        """

        if filename is None:
            raise ValueError('read method require a parameter ``filename``!')

        protocol = kwargs.pop('protocol', None)
        sortbydate = kwargs.pop('sortbydate', True)

        if protocol is None:
            # try to estimate the protocol from the file name extension
            _, extension = os.path.splitext(filename)
            if len(extension) > 0:
                protocol = extension[1:].lower()

        if protocol == 'scp':
            # default reader
            return cls.load(filename, protocol='scp')

            # try:
            # find the adequate reader
        _reader = getattr(cls, 'read_{}'.format(protocol))
        return _reader(filename, sortbydate=sortbydate, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # Generic write function
    # ------------------------------------------------------------------------------------------------------------------

    def write(self, filename, **kwargs):
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
        protocol = kwargs.pop('protocol', None)

        if not protocol:
            # try to estimate the protocol from the file name extension
            _, extension = os.path.splitext(filename)
            if len(extension) > 0:
                protocol = extension[1:].lower()

        if protocol == 'scp':
            return self.save(filename)

        # find the adequate reader

        try:
            # find the adequate reader
            _writer = getattr(self, 'write_{}'.format(protocol))
            return _writer(filename, **kwargs)

        except:

            raise AttributeError('The specified writter '
                                 'for protocol `{}` was not found!'.format(
                protocol))


# make some methods accessible from the main scp API
# ----------------------------------------------------------------------------------------------------------------------

load = NDIO.load
read = NDIO.read
write = NDIO.write

__all__ += ['load', 'read', 'write']

# ======================================================================================================================
if __name__ == '__main__':
    pass
