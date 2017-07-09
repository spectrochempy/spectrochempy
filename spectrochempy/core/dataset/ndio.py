# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import logging
import copy

from traitlets import Unicode, Bool, HasTraits

#from pyface.api import FileDialog, OK

import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import Figure
from numpy.compat import asbytes, asstr, asbytes_nested, bytes, \
    basestring, unicode
from numpy.lib.npyio import zipfile_factory, NpzFile
from numpy.lib.format import write_array, MAGIC_PREFIX

# local import


from spectrochempy.core.dataset.ndaxes import Axes, Axis
from spectrochempy.core.units import Unit

import spectrochempy

#from spectrochempy.preferences.preference_manager import preference_manager as pm
from spectrochempy.logger import log

__all__ = ['NDIO']


class NDIO(HasTraits):
    """
    Import/export interface
    from :class:`~spectrochempy.core.dataset.nddataset.NDDataset`

    This class is used as basic import/export interface of the
    :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.

    """

    def _get_datadir(self):
        """`str`- Default directory for i/o operations.

        """
        datadir = self.preferences.datadir
        return datadir

    def _set_datadir(self, value):
        pass  # preferences_manager.root.datadir = value
        # preferences_manager.preferences.save()


    ## BASIC READER ##
    ##################

    @classmethod
    def load(cls, path='', protocol='SCP'):
        """
        Load a dataset object saved as a pickle file (``.SCP`` file).
        It's a class method, that can be used directly on the class,
        without prior opening of a class instance.

        Parameters
        ----------
        path : `str`
            The path to the file to be read

        kwargs : optional keyword parameters
            Any additional keyword to pass to the actual reader

        Examples
        --------

        >>> from spectrochempy.api import NDDataset
        >>> mydataset = NDDataset.load('DATA/myexperiment.SCP')
        >>> print(mydataset) # doctest: +ELLIPSIS
        <BLANKLINE>
        ...


        Notes
        -----
        adapted from :func:`numpy.load`

        See Also
        --------
        :meth:`read`

        """

        if protocol not in ['SCP']:
            return cls.read(path, protocol=protocol)

        # open file dialog box

        filename = path
        if not path:
            dlg = FileDialog(action='open',
        wildcard='Spectrochempy (*.SCP)|*.SCP|Sappy --DEPRECATED (*.sap)|*.sap')
            if dlg.open() == OK:
                filename = dlg.path
            else:
                return None

        # TODO: file open error handling
        fid = open(filename, 'rb')

        _ZIP_PREFIX = asbytes('PK\x03\x04')
        N = len(MAGIC_PREFIX)
        magic = fid.read(N)
        fid.seek(-N, 1)  # back-up
        if magic.startswith(_ZIP_PREFIX):

            # get zip file
            obj = NpzFile(fid, own_fid=True)

            # interpret
            ndim = obj["data"].ndim
            axes = None
            new = cls()

            for key, val in obj.items():
                if key.startswith('axis'):
                    if not axes:
                        axes = [Axis() for _ in range(ndim)]
                    els = key.split('_')
                    setattr(axes[int(els[1])], "_" + els[2], val)
                elif key == "pars.json":
                    pars = json.loads(asstr(val))
                else:
                    setattr(new, "_" + key, val)
            if axes:
                new.axes = axes

            def setattributes(clss, key, val):
                # utility function to set the attributes
                if key in ['modified', 'date']:
                    val = datetime.datetime.fromtimestamp(val)
                    setattr(clss, "_" + key, val)
                elif key == 'meta':
                    clss.meta.update(val)
                elif key in ['units']:
                    setattr(clss, key, val)
                else:
                    setattr(clss, "_" + key, val)

            for key, val in pars.items():

                if key.startswith('axis'):

                    els = key.split('_')
                    setattributes(axes[int(els[1])], els[2], val)

                else:

                    setattributes(new, key, val)

            return new

        else:
            raise IOError("Failed to load file %s " % filename)
            # finally:
            #    fid.close()

    def save(self, path=''):
        """
        Save the :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
        (default extension: ``.SCP`` ).

        Parameters
        ----------
        path : `str`
            The path to the file to be save

        compress : `bool`
            Whether or not to compress the saved file (default:`True`)

        Notes
        -----
        adapted from :class:`numpy.savez`

        See Also
        ---------
        :meth:`write`

        """

        # open file dialog box
        filename = path

        if not path:
            dlg = FileDialog(action='save as',
                             wildcard='Spectrochempy (*.SCP)|*.SCP')
            if dlg.open() == OK:
                filename = dlg.path
            else:
                return None

        # Import is postponed to here since zipfile depends on gzip, an optional
        # component of the so-called standard library.
        import zipfile
        # Import deferred for startup time improvement
        import tempfile

        if not filename.endswith('.SCP'):
            file = filename + '.SCP'

        compression = zipfile.ZIP_DEFLATED
        zipf = zipfile_factory(filename, mode="w", compression=compression)

        # Stage arrays in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-spectrochempy.tmp')
        os.close(fd)

        pars = {}
        objnames = dir(self)

        def _loop_on_obj(_names, obj=self, level=''):
            """Recursive scan on NDDataset objects"""

            for key in _names:

                val = getattr(obj, key)
                if isinstance(val, np.ndarray):

                    with open(tmpfile, 'wb') as fid:
                        write_array(fid, np.asanyarray(val))
                        zipf.write(tmpfile, arcname=level + key + '.npy')

                elif isinstance(val, Axis):

                    _objnames = dir(val)
                    _loop_on_obj(_objnames, level=key + '.')

                elif isinstance(val, Axes):

                    for i, val in enumerate(val._axes):
                        _objnames = dir(val)
                        _loop_on_obj(_objnames, obj=val, level="axis_%d_" % i)

                elif isinstance(val, datetime.datetime):

                    pars[level + key] = val.timestamp()

                elif isinstance(val, Unit):

                    pars[level + key] = str(val)


                elif isinstance(val, spectrochempy.api.Meta):

                    pars[level + key] = val.to_dict()

                else:

                    pars[level + key] = val

        _loop_on_obj(objnames)

        with open(tmpfile, 'w') as f:
            f.write(json.dumps(pars))

        zipf.write(tmpfile, arcname='pars.json')

        os.remove(tmpfile)

        zipf.close()

    @classmethod
    def read(self, path, **kwargs):
        """
        Generic read function. It's like load a class method.

        Parameters
        ----------
        path : `str`
            The path to the file to be read

        protocol : `str`
            Protocol used for reading. If not provided, the correct protocol
            is evaluated from the file name extension.

        kwargs : optional keyword parameters
            Any additional keyword to pass to the actual reader

        See Also
        --------
        :meth:`load`

        """

        if path is None:
            raise ValueError('read method require a parameter `path`!')

        protocol = kwargs.pop('protocol', None)
        sortbydate = kwargs.pop('sortbydate', True)

        if protocol is None:
            # try to estimate the protocol from the file name extension
            _, extension = os.path.splitext(path)
            if len(extension) > 0:
                protocol = extension[1:].lower()

        if protocol == 'SCP':
            # default reader
            return self.load(path)

        try:
            # find the adequate reader
            _reader = getattr(self, 'read_{}'.format(protocol))
            return _reader(path, protocol='protocol',
                           sortbydate=sortbydate,
                           **kwargs)

        except:
            raise ValueError('The specified importer '
                             'for protocol `{}` was not found!'.format(
                protocol))

    def write(self, path, **kwargs):
        """
        Generic write function which actually delegate the work to an
        writer defined by the parameter ``protocol``.

        Parameters
        ----------
        path : `str`
            The path to the file to be read

        protocol : `str`
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
            _, extension = os.path.splitext(path)
            if len(extension) > 0:
                protocol = extension[1:].lower()

        if protocol == 'SCP':
            return self.save(path)

        # find the adequate reader

        try:
            # find the adequate reader
            _writer = getattr(self, 'write_{}'.format(protocol))
            return _writer(path, protocol='protocol',
                           **kwargs)

        except:

            raise ValueError('The specified writter '
                             'for protocol `{}` was not found!'.format(
                protocol))

    def plot(self,
             ax=None,
             figsize=None,
             fontsize=None,
             kind='modal',
             **kwargs):

        """
        Generic plot function for a
        :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
        which actually delegate
        the work to a plotter defined by the parameter ``kind``.

        Parameters
        ----------
        kind : `str`, optional
            The kind of plot of the dataset,
            which will determine the plotter to use.
            For instance, for 2D data, it can be `contour`, `flat` or `stacked',
            among other kind.

        ax : :class:`matplotlib.Axe` instance. Optional, default = current or new one)
            The axe where to plot

        figsize : `tuple`, optional, default is mpl.rcParams['figure.figsize']
            The figure size

        fontsize : `int`, optional
            The font size in pixels, default is 10 (or read from preferences)

        See Also
        --------
        :meth:`show`

        """

        log.debug('Standard Plot...')

        # -------------------------------------------------------------------------
        # setup figure and axes
        # -------------------------------------------------------------------------

        fig = kwargs.pop('fig', None)
        if isinstance(fig, int):
            if plt.fignum_exists(fig):
                self.fig = plt.figure(fig)
                log.debug('get the existing figure %d' % fig)

            else:  # we need to create a new one
                fig = None

        if fig is None:

            # when using matplotlib inline
            # dpi is the savefig.dpi so we should set it here

            figsize = mpl.rcParams['figure.figsize'] = \
                kwargs.pop('figsize', mpl.rcParams['figure.figsize'])
            fontsize = mpl.rcParams['font.size'] = \
                kwargs.pop('fontsize', mpl.rcParams['font.size'])
            mpl.rcParams['legend.fontsize'] = int(fontsize * .8)
            mpl.rcParams['xtick.labelsize'] = int(fontsize)
            mpl.rcParams['ytick.labelsize'] = int(fontsize)

            self.fig = plt.figure(figsize=figsize, tight_layout=None)
            log.debug('fig is None, create a new figure')

        elif isinstance(fig, Figure):
            log.debug('fig is a figure instance, get this one')
            self.fig = fig

        # for generic plot we assume only a single ax.
        # other plugin class will or are taking care of other needs
        log.debug('get or create a new ax')
        self.ax = self.fig.gca()

        # color cycle
        # prop_cycle = pm.plot.prop_cycle
        # mpl.rcParams['axes.prop_cycle']= r" cycler('color', %s) " % prop_cycle

        # pyplot.subplots_adjust(left=(5 / 25.4) / figure.xsize,
        #                        bottom=(4 / 25.4) / figure.ysize,
        #                        right=1 - (1 / 25.4) / figure.xsize,
        #                        top=1 - (3 / 25.4) / figure.ysize)
        # -------------------------------------------------------------------------
        # select plotter depending on the dimension of the data
        # -------------------------------------------------------------------------
        kind = kwargs.pop('kind', 'generic')

        try:
            # find the adequate plotter
            _plotter = getattr(self, 'plot_{}'.format(kind))
            return _plotter(ax=ax, **kwargs)

        except:
            pass

        try:
            if self.ndim == 1:
                self.plot1D(ax=ax, **kwargs)
            elif self.ndim == 2:
                self.plot2D(ax=ax, **kwargs)
            elif self.ndim == 3:
                raise ValueError  # self.plot3D(ax=ax, **kwargs)

        except:

            raise ValueError('The specified plotter '
                             'for kind `{}` was not found!'.format(kind))

        # -------------------------------------------------------------------------
        # additional matplotlib commands
        # -------------------------------------------------------------------------
        commands = kwargs.get('commands', [])
        if commands:
            for command in commands:
                com, val = command.split('(')
                val = val.split(')')[0].split(',')
                ags = []
                kws = {}
                for item in val:
                    if '=' in item:
                        k, v = item.split('=')
                        kws[k.strip()] = eval(v)
                    else:
                        ags.append(eval(item))
                getattr(self.ax, com)(*ags, **kws)

        # adjust the plots
        # subplot dimensions
        top = kwargs.pop('top', mpl.rcParams['figure.subplot.top'])
        bottom = kwargs.pop('bottom', mpl.rcParams['figure.subplot.bottom'])
        left = kwargs.pop('left', mpl.rcParams['figure.subplot.left'])
        right = kwargs.pop('right', mpl.rcParams['figure.subplot.right'])

        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        # self.fig.tight_layout()

        # finally return the current fig for further manipulation.

        # should be after all plot commands
        savename = kwargs.get('savefig', None)
        if savename is not None:
            self.fig.savefig(savename)



    def show(self):
        """
        Method to force the `matplotlib` figure display

        See Also
        --------
        :meth:`plot`

        """
        if hasattr(self, 'fig'):
            plt.show

    def __getstate__(self):
        # needed to remove some entry to avoid picling them
        state = super(NDIO, self).__getstate__()

        for key in self._all_func_names:
            if key in state:
                del state[key]

        statekeys = list(state.keys())
        for key in statekeys:
            if not key.startswith('_'):
                del state[key]

        statekeys = list(state.keys())
        for key in statekeys:
            if key.startswith('__'):
                del state[key]

        return state


# =============================================================================
# Modify the doc to include Traits
# =============================================================================
#from spectrochempy.utils import create_traitsdoc

#create_traitsdoc(NDIO)
