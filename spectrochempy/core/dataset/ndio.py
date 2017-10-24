# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
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

import os
import copy
import logging
import json
import datetime
import warnings

from traitlets import Unicode, Bool, HasTraits, Instance, observe, default

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import isinteractive, Figure, Axes as Ax
# change the name to avoid
# collisions with
# spectrochempy Axes objets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectrochempy.utils import is_kernel

import numpy as np
from numpy.compat import asbytes, asstr
from numpy.lib.npyio import zipfile_factory, NpzFile
from numpy.lib.format import write_array, MAGIC_PREFIX

# local import
# ------------

import spectrochempy
from spectrochempy.core.dataset.ndcoords import CoordSet, Coord
from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.units import Unit
from spectrochempy.utils import is_sequence
from spectrochempy.gui import gui
from spectrochempy.application import plotoptions
from spectrochempy.application import options

# Constants
# ---------

__all__ = ['NDIO',

           'curfig',
           'show',
           'figure',

           'plot',
           'load',
           'read',
           'write',

           'available_styles'

           ]
_classes = ['NDIO']

from spectrochempy.application import log


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

    _ax = Instance(Ax, allow_none=True)
    _fig = Instance(Figure, allow_none=True)

    # --------------------------------------------------------------------------
    # Generic save function
    # --------------------------------------------------------------------------

    def save(self, path='',
             **kwargs
             ):
        """
        Save the :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
        (default extension: ``.scp`` ).

        Parameters
        ----------
        path : `str`
            The path to the file to be save

        directory : `str` [optional, default=`True`]
            It specified, the given path (generally a file name) fill be
            appended to the ``dir``.

        Examples
        ---------
        Read some experimental data and then save in our proprietary format **scp**

        >>> from spectrochempy.api import NDDataset, data
        >>> mydataset = NDDataset.read_omnic('irdata/NH4Y-activation.SPG', directory=data)
        >>> mydataset.save('mydataset.scp', directory=data)

        Notes
        -----
        adapted from :class:`numpy.savez`

        See Also
        ---------
        write

        """

        # open file dialog box
        filename = path

        if not path:
            filename = gui.saveFileDialog()

        if not filename:
            raise IOError('no filename provided!')

        if not filename.endswith('.scp'):
            filename = filename + '.scp'

        directory = kwargs.get("directory", options.data)
        if not os.path.exists(directory):
            raise IOError("directory doesn't exists!")

        if os.path.isdir(directory):
            filename = os.path.expanduser(os.path.join(directory, filename))
        else:
            warnings.warn('Provided directory is a file, '
                          'so we use its parent directory')
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

                val = getattr(obj, key)
                if isinstance(val, np.ndarray):

                    with open(tmpfile, 'wb') as fid:
                        write_array(fid, np.asanyarray(val))
                        zipf.write(tmpfile, arcname=level + key + '.npy')

                elif isinstance(val, Coord):

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


                elif isinstance(val, Meta):

                    pars[level + key] = val.to_dict()

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
             path='',
             protocol='scp',
             **kwargs
             ):
        """
        Load a dataset object saved as a pickle file ( ``.scp`` file).
        It's a class method, that can be used directly on the class,
        without prior opening of a class instance.

        Parameters
        ----------
        path : `str`

            The path to the file to be read.

        protocol : `str`

            optional, default= ``scp``
            The default type for saving,

        directory : `str`

            optional, default=``data``
            The directory from where to load hhe file.

        kwargs : optional keyword parameters.

            Any additional keyword to pass to the actual reader.

        Examples
        --------

        >>> from spectrochempy.api import NDDataset,data
        >>> mydataset = NDDataset.load('mydataset.scp', directory=data)
        >>> print(mydataset)                  # doctest: +ELLIPSIS
        <BLANKLINE>
        ...

        by default, directory for saving is the `data`.
        So the same thing can be done simply by:

        >>> from spectrochempy.api import NDDataset,data
        >>> mydataset = NDDataset.load('mydataset.scp')
        >>> print(mydataset)                  # doctest: +ELLIPSIS
        <BLANKLINE>
        ...


        Notes
        -----
        adapted from `numpy.load`

        See Also
        --------
        read, save


        """

        if protocol not in ['scp']:
            return cls.read(path, protocol=protocol)

        # open file dialog box

        filename = path
        directory = kwargs.get("directory", options.data)
        if not path:
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
            obj = NpzFile(fid, own_fid=True)

            # interpret
            ndim = obj["data"].ndim
            axes = None
            new = cls()

            for key, val in obj.items():
                if key.startswith('axis'):
                    if not axes:
                        axes = [Coord() for _ in range(ndim)]
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

    # --------------------------------------------------------------------------
    # Generic read function
    # --------------------------------------------------------------------------
    @classmethod
    def read(cls, path, **kwargs):
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

        if protocol == 'scp':
            # default reader
            return cls.load(path)

        try:
            # find the adequate reader
            _reader = getattr(cls, 'read_{}'.format(protocol))
            return _reader(path, protocol='protocol',
                           sortbydate=sortbydate,
                           **kwargs)

        except:
            raise ValueError('The specified importer '
                             'for protocol `{}` was not found!'.format(
                    protocol))

    # --------------------------------------------------------------------------
    # Generic write function
    # --------------------------------------------------------------------------

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

        if protocol == 'scp':
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

    # --------------------------------------------------------------------------
    # generic plotter and plot related methods or properties
    # --------------------------------------------------------------------------

    def plot(self, **kwargs):

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
            For instance, for 2D data, it can be `map`, `stack' or 'image'
            among other kind.

        ax : :class:`matplotlib.Axe` instance. Optional, default = current or new one)
            The axe where to plot

        figsize : `tuple`, optional, default is mpl.rcParams['figure.figsize']
            The figure size

        fontsize : `int`, optional
            The font size in pixels, default is 10 (or read from preferences)

        hold : `bool`, optional, default = `False`.

            Should we plot on the ax previously used
            or create a new figure?

        style : `str`

        See Also
        --------
        :meth:`show`

        """

        log.debug('Standard Plot...')

        # color cycle
        # prop_cycle = options.prop_cycle
        # mpl.rcParams['axes.prop_cycle']= r" cycler('color', %s) " % prop_cycle

        # -------------------------------------------------------------------------
        # select plotter depending on the dimension of the data
        # -------------------------------------------------------------------------
        kind = kwargs.pop('kind', 'generic')

        # Find or guess the adequate plotter
        # -----------------------------------

        try:
            _plotter = getattr(self, 'plot_{}'.format(kind))

        except:  # no plotter found
            log.error('The specified plotter '
                      'for kind `{}` was not found!'.format(kind))
            return None

        # Execute the plotter
        # --------------------

        if not _plotter(**kwargs):
            return None

        return self._ax

    # --------------------------------------------------------------------------
    # setup figure properties
    # --------------------------------------------------------------------------
    def _figure_setup(self, ndim=1, **kwargs):

        # set temporarity a new style if any
        # ----------------------------------
        plt.style.use('classic')
        plt.style.use(plotoptions.style)
        style = kwargs.pop('style', None)

        if style:
            if not is_sequence(style):
                style = [style]
            if isinstance(style, dict):
                style = [style]
            style = [plotoptions.style] + list(style)
            plt.style.use(style)

        # size of the figure and other properties
        # ---------------------------------------
        self._figsize = mpl.rcParams['figure.figsize'] = \
            kwargs.pop('figsize', mpl.rcParams['figure.figsize'])

        fontsize = mpl.rcParams['font.size'] = \
            kwargs.pop('fontsize', mpl.rcParams['font.size'])
        mpl.rcParams['legend.fontsize'] = int(fontsize * .8)
        mpl.rcParams['xtick.labelsize'] = int(fontsize)
        mpl.rcParams['ytick.labelsize'] = int(fontsize)

        # Get current figure information
        # ------------------------------
        # if (self._fig is None and self._ax is None):
        if curfig() is None:
            self._updateplot = False  # the figure doesn't yet exists.
            self._fignum = kwargs.pop('fignum', None)  # self._fignum)
            # if no figure present, then create one with the fignum number
            self._fig = plt.figure(self._fignum, figsize=self._figsize)
            self._ax = self._fig.gca()
        else:
            self._updateplot = True  # fig exist: updateplot
            self._fig = curfig()
            if ndim > 1 and self._fig.get_axes():
                self._ax, self._axec = self._fig.get_axes()

            else:
                self._ax = self._fig.gca()

        # elif self._ax is not None:
        #    self._fig = fig = self._ax.figure

        # Get the number of the present figure
        self._fignum = self._fig.number

        if ndim == 2:
            # TODO: also the case of 3D

            # show projections (only useful for maps)
            # ---------------------------------------

            colorbar = kwargs.get('colorbar', True)

            proj = kwargs.get('proj', plotoptions.show_projections)
            # TODO: tell the axis by title.

            xproj = kwargs.get('xproj', plotoptions.show_projection_x)

            yproj = kwargs.get('yproj', plotoptions.show_projection_y)

            kind = kwargs.get('kind', plotoptions.kind_2D)

        # for generic plot, we assume only a single axe with possible projections
        # and colobar
        #
        # other plot class may take care of other needs

        ax = self._ax

        if ndim == 2 and kind in ['map', 'image'] and self._divider is None:
            # create new axes on the right and on the top of the current axes
            # The first argument of the new_vertical(new_horizontal) method is
            # the height (width) of the axes to be created in inches.
            #
            # This is necessary for projections and colorbar

            self._divider = divider = make_axes_locatable(ax)
            # print divider.append_axes.__doc__

            if proj or xproj:
                axex = divider.append_axes("top", 1.01, pad=0.01, sharex=ax,
                                           frameon=0, yticks=[])
                axex.tick_params(bottom='off', top='off')
                plt.setp(axex.get_xticklabels() + axex.get_yticklabels(),
                         visible=False)
                self._axex = axex

            if proj or yproj:
                axey = divider.append_axes("right", 1.01, pad=0.01, sharey=ax,
                                           frameon=0, xticks=[])
                axey.tick_params(right='off', left='off')
                plt.setp(axey.get_xticklabels() + axey.get_yticklabels(),
                         visible=False)
                self._axey = axey

            if colorbar:
                axec = divider.append_axes("right", .15, pad=0.3, frameon=0,
                                           xticks=[], yticks=[])
                axec.tick_params(right='off', left='off')
                # plt.setp(axec.get_xticklabels(), visible=False)

                self._axec = axec

        return

    # --------------------------------------------------------------------------
    # resume the plot
    # --------------------------------------------------------------------------
    def _plot_resume(self, **kwargs):

        # Additional matplotlib commands on the current plot
        # ----------------------------------------------------------------------

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
        # top = kwargs.pop('top', mpl.rcParams['figure.subplot.top'])
        # bottom = kwargs.pop('bottom', mpl.rcParams['figure.subplot.bottom'])
        # left = kwargs.pop('left', mpl.rcParams['figure.subplot.left'])
        # right = kwargs.pop('right', mpl.rcParams['figure.subplot.right'])

        # plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        # self.fig.tight_layout()

        # finally return the current fig for further manipulation.

        # should be after all plot commands
        savename = kwargs.get('savefig', None)
        if savename is not None:
            self._fig.savefig(savename)

        # if not plotoptions.do_not_block:
        plt.draw()

    def plot_generic(self, **kwargs):
        """
        The generic plotter. It try to guess an adequate basic plot for the data

        Parameters
        ----------
        ax : :class:`matplotlib.axe'

            the viewplot where to plot.

        kwargs : optional additional arguments

        Returns
        -------

        """

        # reduce 2D data with  only one row to 1D
        # the same for ND that must be reduce to the minimal form.

        temp = self.squeeze()  # create a copy by default while squeezing

        if temp.ndim == 1:

            temp.plot_1D(**kwargs)

        elif temp.ndim == 2:

            temp.plot_2D(**kwargs)

        elif temp.ndim == 3:

            temp.plot_3D(**kwargs)

        else:
            log.error('Cannot guess an adequate plotter. I did nothing!')
            return False

        self._ax = temp._ax
        self._axec = temp._axec
        self._axex = temp._axex
        self._axey = temp._axey
        self._fig = temp._fig
        self._fignum = temp._fignum

        return True  # Everything was OK

    # -------------------------------------------------------------------------
    # Special attributes
    # -------------------------------------------------------------------------

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

    def __dir__(self):
        return ['fignum', 'ax', 'axec', 'axex', 'axey', 'divider']

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def fig(self):
        """
        Matplotlib figure associated to this dataset

        """
        return self._fig

    @property
    def fignum(self):
        """
        Matplotlib figure associated to this dataset

        """
        return self._fignum

    @property
    def ax(self):
        """
        Matplotlib axe associated to this dataset

        """
        if self._ax is None:
            self._ax = self._fig.gca()
        return self._ax

    @property
    def axec(self):
        """
        Matplotlib colorbar axe associated to this dataset

        """
        return self._axec

    @property
    def axex(self):
        """
        Matplotlib projection x axe associated to this dataset

        """
        return self._axex

    @property
    def axey(self):
        """
        Matplotlib projection y axe associated to this dataset

        """
        return self._axey

    @property
    def divider(self):
        """
        Matplotlib plot divider

        """
        return self._divider


def curfig():
    n = plt.get_fignums()
    if not n:
        return None
    fig = plt.gcf()
    return fig


def show():
    """
    Method to force the `matplotlib` figure display

    """
    if not plotoptions.do_not_block or isinteractive():
        if curfig():
            plt.show()


def available_styles():
    return ['notebook', 'paper', 'poster', 'talk', 'sans']


figure = plt.figure
plot = NDIO.plot
load = NDIO.load
read = NDIO.read
write = NDIO.write
