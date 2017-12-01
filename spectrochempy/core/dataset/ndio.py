# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

"""
This module define the class :class:`NDIO` in which input/output standard
methods for a :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
are defined.

"""

# Python and third parties imports
# ----------------------------------

import datetime
import time
import json
import os

import numpy as np
from numpy.compat import asbytes, asstr
from numpy.lib.format import write_array, MAGIC_PREFIX
from numpy.lib.npyio import zipfile_factory, NpzFile
from traitlets import Dict, List, Float, HasTraits, Instance, observe, All
import matplotlib.pyplot as plt

# local import
# ------------
from spectrochempy.core.dataset.ndcoords import Coord, CoordSet
from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.units.units import Unit
from spectrochempy.gui import gui
from spectrochempy.utils import SpectroChemPyWarning
from spectrochempy.utils import is_sequence
from spectrochempy.application import app
plotoptions = app.plotoptions

log = app.log
options = app

# Constants
# ---------

__all__ = ['NDIO',

           'load',
           'read',
           'write',

           ]

_classes = ['NDIO']


# ==============================================================================
# Class NDIO to handle I/O of datasets
# ==============================================================================

class NDIO(HasTraits):
    """
    Import/export interface
    from :class:`~spectrochempy.core.dataset.nddataset.NDDataset`

    This class is used as basic import/export interface of the
    :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.

    """

    # --------------------------------------------------------------------------
    # Generic save function
    # --------------------------------------------------------------------------

    def save(self, filename='',
             **kwargs
             ):
        """
        Save the :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
        (default extension: ``.scp`` ).

        Parameters
        ----------

        path : str

            The filename to the file to be save

        directory : str [optional, default = `True`]

            It specified, the given filename (generally a file name) fill be
            appended to the ``dir``.

        Examples
        ---------

        Read some experimental data and then save in our proprietary format **scp**

        >>> from spectrochempy.api import NDDataset, scpdata # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        <BLANKLINE>
            SpectroChemPy's API
            Version   : 0.1...
        >>> mydataset = NDDataset.read_omnic('irdata/NH4Y-activation.SPG', directory=scpdata)
        >>> mydataset.save('mydataset.scp', directory=scpdata)

        Notes
        -----
        adapted from :class:`numpy.savez`

        See Also
        ---------

        :meth:`write`

        """

        # open file dialog box
        filename = filename

        if not filename:
            filename = gui.saveFileDialog()

        if not filename:
            raise IOError('no filename provided!')

        if not filename.endswith('.scp'):
            filename = filename + '.scp'

        directory = kwargs.get("directory", options.scpdata)
        if not os.path.exists(directory):
            raise IOError("directory doesn't exists!")

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

        # Stage arrays in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-spectrochempy.tmp')
        os.close(fd)

        pars = {}
        objnames = dir(self)

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

                elif isinstance(val, Unit):

                    pars[level + key] = str(val)

                elif isinstance(val, Meta):

                    pars[level + key] = val.to_dict()

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

    # --------------------------------------------------------------------------
    # Generic load function
    # --------------------------------------------------------------------------

    @classmethod
    def load(cls,
             filename='',
             protocol='scp',
             **kwargs
             ):
        """Load a dataset object saved as a pickle file ( ``.scp`` file).
        It's a class method, that can be used directly on the class,
        without prior opening of a class instance.

        Parameters
        ----------

        path : str

            The filename to the file to be read.

        protocol : str

            optional, default= ``scp``
            The default type for saving,

        directory : str

            optional, default= ``data``
            The directory from where to load the file.

        kwargs : optional keyword parameters.

            Any additional keyword to pass to the actual reader.

        Examples
        --------

        >>> from spectrochempy.api import NDDataset,scpdata
        >>> mydataset = NDDataset.load('mydataset.scp', directory=scpdata)
        >>> print(mydataset)                  # doctest: +ELLIPSIS
        <BLANKLINE>
        ...

        by default, directory for saving is the `data`.
        So the same thing can be done simply by:

        >>> from spectrochempy.api import NDDataset,scpdata
        >>> mydataset = NDDataset.load('mydataset.scp')
        >>> print(mydataset)                  # doctest: +ELLIPSIS
        <BLANKLINE>
        ...


        Notes
        -----

        adapted from `numpy.load`

        See Also
        --------

        :meth:`read`, :meth:`save`


        """

        if protocol not in ['scp']:
            return cls.read(filename, protocol=protocol)

        # open file dialog box

        directory = kwargs.get("directory", options.scpdata)
        if not filename:
            filename = gui.openFileNameDialog(directory=directory)
            if not filename:
                raise IOError('no filename provided!')
        else:
            try:
                if not directory:
                    fid = open(filename, 'rb')
                else:
                    # cast to  file in the testdata directory
                    # TODO: add possibility to search in several directory
                    fid = open(
                            os.path.expanduser(
                                    os.path.join(directory, filename)),
                            'rb')
            except:
                raise IOError('no valid filename provided')

        _ZIP_PREFIX = asbytes('PK\x03\x04')
        N = len(MAGIC_PREFIX)
        magic = fid.read(N)
        fid.seek(-N, 1)  # back-up
        if magic.startswith(_ZIP_PREFIX):

            # get zip file
            obj = NpzFile(fid, allow_pickle=True)

            # interpret
            ndim = obj["data"].ndim
            coordset = None
            new = cls()

            for key, val in list(obj.items()):
                if key.startswith('coord_'):
                    if not coordset:
                        coordset = [Coord() for _ in range(ndim)]
                    els = key.split('_')
                    setattr(coordset[int(els[1])], "_%s" % els[2], val)
                elif key == "pars.json":
                    pars = json.loads(asstr(val))
                else:
                    setattr(new, "_%s" % key, val)
            if coordset:
                new.coordset = coordset

            def setattributes(clss, key, val):
                # utility function to set the attributes
                if key in ['modified', 'date']:
                    val = datetime.datetime.fromtimestamp(val)
                    setattr(clss, "_%s" % key, val)
                elif key == 'meta':
                    clss.meta.update(val)
                elif key in ['units']:
                    setattr(clss, key, val)
                else:
                    setattr(clss, "_%s" % key, val)

            for key, val in list(pars.items()):

                if key.startswith('coord_'):

                    els = key.split('_')
                    setattributes(coordset[int(els[1])], els[2], val)

                else:

                    setattributes(new, key, val)

            return new

        else:
            raise IOError("Failed to load file %s " % filename)
            # finally:
            #    fid.close()

    # --------------------------------------------------------------------------
    # Generic read function
    # --------------------------------------------------------------------------

    @classmethod
    def read(cls,
             filename=None, **kwargs):
        """
        Generic read function. It's like load a class method.

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

        :meth:`load`

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
            return cls.load(filename)

            # try:
            # find the adequate reader
        _reader = getattr(cls, 'read_{}'.format(protocol))
        return _reader(filename, protocol='protocol',
                       sortbydate=sortbydate,
                       **kwargs)

    # --------------------------------------------------------------------------
    # Generic write function
    # --------------------------------------------------------------------------

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
            :class:`~spectrochempy.core.dataset.nddataset.NDDataset` in a file,
            which will determine the exporter to use.

        kwargs : optional keyword parameters

            Any additional keyword to pass to the actual exporter

        See Also
        --------

        :meth:`save`

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
            return _writer(filename, protocol='protocol',
                           **kwargs)

        except:

            raise AttributeError('The specified writter '
                                 'for protocol `{}` was not found!'.format(
                    protocol))

    # -------------------------------------------------------------------------
    # Special attributes
    # -------------------------------------------------------------------------

    # def __getstate__(self):   #TODO: not sure still have needs for that. CHECK!
    #     # needed to remove some entry to avoid pickling them
    #     state = super(NDIO, self).__getstate__()
    #
    #     for key in self._all_func_names:
    #         if key in state:
    #             del state[key]
    #
    #     statekeys = list(state.keys())
    #     for key in statekeys:
    #         if not key.startswith('_'):
    #             del state[key]
    #
    #     statekeys = list(state.keys())
    #     for key in statekeys:
    #         if key.startswith('__'):
    #             del state[key]
    #
    #     return state


load = NDIO.load
read = NDIO.read
write = NDIO.write

if __name__ == '__main__':

    # test interactive masks

    from spectrochempy.api import *

    A = NDDataset.read_omnic(
                os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))
    A.y -= A.y[0]
    A.y.to('hour', inplace=True)
    A.y.title = u'Aquisition time'
    ax = A[:,1600.:4000.].plot_stack(y_showed = [2.,6.])

    def _test_save():

        A.save('essai')
        A.save('essai2')

        os.remove(os.path.join(scpdata, 'essai.scp'))
        os.remove(os.path.join(scpdata, 'essai2.scp'))

    _test_save()
    show()


