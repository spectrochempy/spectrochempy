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

import datetime
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from numpy.compat import asbytes, asstr
from numpy.lib.format import write_array, MAGIC_PREFIX
from numpy.lib.npyio import zipfile_factory, NpzFile
from traitlets import Dict, List, HasTraits, Instance


# local import
# ------------

from spectrochempy.core.dataset.ndarray import CoordSet, masked
from spectrochempy.core.dataset.ndcoords import Coord
from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.units import Unit
from spectrochempy.gui import gui
from spectrochempy.utils import SpectroChemPyWarning
from spectrochempy.utils import is_sequence
from spectrochempy.application import plotoptions, log, options

# Constants
# ---------

__all__ = ['NDIO',

           'curfig',
           'show',

           'plot',
           'load',
           'read',
           'write',

           # 'interactive_masks',
           'set_figure_style',
           'available_styles'

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

    # The figure on which this dataset can be plotted
    _fig = Instance(plt.Figure, allow_none=True)

    # The axes on which this dataset and other elements such as projections 
    # and colorbar can be plotted
    _axes = Dict(Instance(plt.Axes))

    _selected = List()

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

        path : `str`

            The filename to the file to be save

        directory : `str` [optional, default = `True`]

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

        path : `str`

            The filename to the file to be read.

        protocol : `str`

            optional, default= ``scp``
            The default type for saving,

        directory : `str`

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
        filename : `str`

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

        filename : `str`

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

    # --------------------------------------------------------------------------
    # generic plotter and plot related methods or properties
    # --------------------------------------------------------------------------

    _general_parameters_doc_ = """
    
savefig: `str`

    A string containing a path to a filename. The output format is deduced 
    from the extension of the filename. If the filename has no extension, 
    the value of the rc parameter savefig.format is used.

dpi : [ None | scalar > 0]

    The resolution in dots per inch. If None it will default to the 
    value savefig.dpi in the matplotlibrc file.
    
    """

    def plot(self, **kwargs):

        """
        Generic plot function for
        a :class:`~spectrochempy.core.dataset.nddataset.NDDataset` which
        actually delegate the work to a plotter defined by the parameter ``kind``.

        Parameters
        ----------

        kind : `str`, optional

            The kind of plot of the dataset,
            which will determine the plotter to use.
            For instance, for 2D data, it can be `map`, `stack` or `image`
            among other kind.

        ax : :class:`matplotlib.Axes` instance. Optional, default = current or new one

            The axe where to plot

        figsize : `tuple`, optional, default is mpl.rcParams['figure.figsize']

            The figure size

        fontsize : `int`, optional

            The font size in pixels, default is 10 (or read from preferences)

        hold : `bool`, optional, default = `False`.

            Should we plot on the ax previously used
            or create a new figure?

        style : `str`

        autolayout : `bool`, optional, default=``True``

            if True, layout will be set automatically

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
            raise IOError('The specified plotter '
                          'for kind `{}` was not found!'.format(kind))

        # Execute the plotter
        # --------------------

        _plotter(**kwargs)

    # --------------------------------------------------------------------------
    # setup figure properties
    # --------------------------------------------------------------------------

    def _figure_setup(self, ndim=1, **kwargs):

        set_figure_style(**kwargs)

        self._figsize = mpl.rcParams['figure.figsize'] = \
            kwargs.get('figsize', mpl.rcParams['figure.figsize'])

        mpl.rcParams[
            'figure.autolayout'] = kwargs.pop('autolayout', True)

        # Get current figure information
        # ------------------------------
        # if curfig() is None:
        #     self._updateplot = False  # the figure doesn't yet exists.
        #     self._fignum = kwargs.pop('fignum', None)  # self._fignum)
        #     # if no figure present, then create one with the fignum number
        #     self._fig = plt.figure(self._fignum, figsize=self._figsize)
        #     self.axes['main'] = self._fig.gca()
        # else:
        log.debug('update plot')
        # self._updateplot = True  # fig exist: updateplot

        # get the current figure
        hold = kwargs.get('hold', False)
        self._fig = curfig(hold)

        # is ax in the keywords ?
        ax = kwargs.pop('ax', None)
        if not hold:
            self._axes = {}  # reset axes
            self._divider = None

        if ax is not None:
            # in this case we will plot on this ax
            if isinstance(ax, plt.Axes):
                ax.name = 'main'
                self.axes['main'] = ax
            # elif isinstance(ax, str) and ax in self.axes.keys():
            #     # next plot commands will be applied if possible to this ax
            #     self._axdest = ax
            # elif isinstance(ax, int) and ax>0 and ax <= len(self.axes.keys()):
            #     # next plot commands will be applied if possible to this ax
            #     ax = "axe%d"%ax
            #     self._axdest = ax
            else:
                raise ValueError('{} is not recognized'.format(ax))

        elif self._fig.get_axes():
            # no ax parameters in keywords, so we need to get those existing
            # We assume that the existing axes have a name
            self.axes = self._fig.get_axes()
        else:
            # or create a new subplot
            ax = self._fig.gca()
            ax.name = 'main'
            self.axes['main'] = ax

        # Get the number of the present figure
        self._fignum = self._fig.number

        # for generic plot, we assume only a single axe with possible projections
        # and colobar
        #
        # other plot class may take care of other needs

        ax = self.axes['main']

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

            SHOWXPROJ = (proj or xproj) and kind in ['map', 'image']
            SHOWYPROJ = (proj or yproj) and kind in ['map', 'image']

            # create new axes on the right and on the top of the current axes
            # The first argument of the new_vertical(new_horizontal) method is
            # the height (width) of the axes to be created in inches.
            #
            # This is necessary for projections and colorbar

            if (SHOWXPROJ or SHOWYPROJ or colorbar) and self._divider is None:
                self._divider = make_axes_locatable(ax)

            divider = self._divider

            if SHOWXPROJ:
                axex = divider.append_axes("top", 1.01, pad=0.01, sharex=ax,
                                           frameon=0, yticks=[])
                axex.tick_params(bottom='off', top='off')
                plt.setp(axex.get_xticklabels() + axex.get_yticklabels(),
                         visible=False)
                axex.name = 'xproj'

            if SHOWYPROJ:
                axey = divider.append_axes("right", 1.01, pad=0.01, sharey=ax,
                                           frameon=0, xticks=[])
                axey.tick_params(right='off', left='off')
                plt.setp(axey.get_xticklabels() + axey.get_yticklabels(),
                         visible=False)
                axey.name = 'yproj'
                self.axes['yproj'] = axey

            if colorbar:
                axec = divider.append_axes("right", .15, pad=0.3, frameon=0,
                                           xticks=[], yticks=[])
                axec.tick_params(right='off', left='off')
                # plt.setp(axec.get_xticklabels(), visible=False)

                axec.name = 'colorbar'
                self.axes['colorbar'] = axec

    # --------------------------------------------------------------------------
    # resume a figure plot
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
                getattr(self.axes['main'], com)(*ags,
                                                **kws)  # TODO:improve this

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
            # we save the figure with options found in kwargs
            # starting with `save`

            kw = {}
            for key, value in kwargs.items():
                if key.startswith('save'):
                    key = key[4:]
                    kw[key] = value
            self._fig.savefig(savename, **kw)

        plt.draw()

        cid = self._fig.canvas.mpl_connect(
                'button_press_event', NDIO._onclick)

    # --------------------------------------------------------------------------
    # plotter: plot_generic
    # --------------------------------------------------------------------------

    def plot_generic(self, **kwargs):
        """
        The generic plotter. It try to guess an adequate basic plot for the data.
        Other kind of plotters are defined explicitely in the `viewer` package.

        Parameters
        ----------

        ax : :class:`matplotlib.axe`

            the viewplot where to plot.

        kwargs : optional additional arguments

        Returns
        -------

        """

        temp = self.copy()

        if temp.ndim == 1:

            temp.plot_1D(**kwargs)

        elif temp.ndim == 2:

            temp.plot_2D(**kwargs)

        elif temp.ndim == 3:

            temp.plot_3D(**kwargs)

        else:
            log.error('Cannot guess an adequate plotter. I did nothing!')
            return False

        self._axes = temp._axes
        self._fig = temp._fig
        self._fignum = temp._fignum


    # --------------------------------------------------------------------------
    # interactive functions
    # --------------------------------------------------------------------------

    def interactive_masks(self, **kwargs):

        # TODO: make it for 1D too!

        kwargs.pop('kind', None)
        self.plot_stack(**kwargs)

        fig = self.fig
        ax = self.axes['main']
        ax.set_title(
                'INTERACTIVE MASK SELECTION (press `h` for help)',
                fontsize='12', color='red')

        _message = \
""" ============================================
 HELP
 ============================================

 --------- KEYS -----------------------------
 h: this help
 u: undo the last set mask.
 x: to apply mask selections and exit. 
 esc : to hide displayed window help
 
 --------- MOUSE ----------------------------
 * click left button to pick a single row
 * click right button to pick a single column
 * click right button and move for a range 
   selection 

 ============================================
"""

        self._helptext = ax.text(0.02, 0.02, _message, fontsize=10,
                                 transform=fig.transFigure, color='blue',
                                 bbox={'facecolor': 'white',
                                       'edgecolor': 'blue'})

        _messclick = 'Click on a line\n' \
                     'with the left button\n' \
                     'to mask a row.\n' \
                     'Click and/or span \n' \
                     'with the right \n' \
                     'or middle button\n' \
                     'for column(s) selection.\n' \
                     'Press `u` to undo\n' \
                     'the last selection mask.' \
                     'Press `x` to exit \n' \
                     'and apply masks.\n' \
                     '`esc` to hide this help.'

        self._tpos = ax.text(0.01, 0.55, _messclick, fontsize=10,
                             transform=fig.transFigure, color='green',
                             bbox={'facecolor': 'white',
                                   'edgecolor': 'green'})

        self._helptext.set_visible(False)
        self._tpos.set_visible(False)

        self._selected = []

        # mouse events
        # ------------
        def _onclick(event):
            # fired on a mouse click.

            # if it is nopt fired in a given axe, return
            # immediately and do nothing
            self._helptext.set_visible(False)
            self._tpos.set_visible(False)

            if event.inaxes.name not in ['main', 'xproj', 'yproj', 'colorbar']:
                return

            if event.button == 3:
                self._tpos.set_visible(False)
                x = event.xdata
                axv = ax.axvline(x, lw=.1, color='white')
                self._selected.append(('col', axv, x))
            else:
                self._tpos.set_visible(True)

            plt.draw()

            pass

        self.fig.canvas.mpl_connect('button_press_event', _onclick)

        # key events
        # ----------

        def _on_key(event):
            self._tpos.set_visible(False)
            if event.key == 'h':
                # we show the help.
                self._helptext.set_visible(True)
                plt.draw()


        def _on_key_release(event):
            self._helptext.set_visible(False)
            self._tpos.set_visible(False)

            if event.key in ['u']:
                if self._selected:
                    last = list(self._selected.pop(-1))
                    if last[0] in ['span','col']:
                        last[1].remove()
                    else:
                        last[1].set_color(last[3])
                        last[1].set_linewidth(last[4])

            if event.key == 'esc':
                self._helptext.set_visible(False)
                self._tpos.set_visible(False)

            if event.key in ['x']:
                log.info("apply all selected mask")

                for item in self._selected:
                    _item = list(item)
                    if _item[0] in ['span']:
                        xmin, xmax = _item[2:]
                        self[:, xmin:xmax]=masked
                    elif _item[0] in ['col']:
                        x = _item[2]
                        self[:, x] = masked
                    elif _item[0] in ['row']:
                        y = int(_item[2])
                        self[y] = masked


                plt.close(self._fig)

            plt.draw()

        self.fig.canvas.mpl_connect('key_press_event', _on_key)
        self.fig.canvas.mpl_connect('key_release_event', _on_key_release)

        # pick event
        # ----------

        def _onpick(event):

            if isinstance(event.artist, Line2D):
                button = event.mouseevent.button
                sel = event.artist
                y = sel.get_label()
                x = event.mouseevent.xdata
                if button == 1:
                    # left button -> row selection
                    color = sel.get_color()
                    lw = sel.get_linewidth()
                    # save these setting to undo
                    self._selected.append(('row', sel, y, color, lw))
                    sel.set_color('gray')
                    sel.set_linewidth(.1)
                elif button == 3:
                    # right button -> column selection
                    axv = ax.axvline(x, lw= .1, color='white')
                    self._selected.append(('col', axv, x))
                self._tpos.set_visible(False)

            plt.draw()

        self.fig.canvas.mpl_connect('pick_event', _onpick)

        def _onspan(xmin, xmax):
            xmin, xmax = sorted((xmin, xmax))
            sp = ax.axvspan(xmin, xmax, facecolor='white',
                            edgecolor='white', alpha=.95,
                            zorder=10000)
            self._selected.append(('span', sp, xmin, xmax))
            self._tpos.set_visible(False)
            plt.draw()

        span = SpanSelector(ax, _onspan, 'horizontal',  minspan=1, button=[2,3],
                            useblit=True, rectprops=dict(alpha=0.95,
                                                         facecolor='white',
                                                         edgecolor='w'))

        show()
        return span

    # -------------------------------------------------------------------------
    # Special attributes
    # -------------------------------------------------------------------------

    def __getstate__(self):
        # needed to remove some entry to avoid pickling them
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
        return ['fignum', 'axes', 'divider']

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
    def axes(self):
        """
        A dictionary containing all the axes of the current figures
        """
        return self._axes

    @axes.setter
    def axes(self, axes):
        # we assume that the axes have a name
        if isinstance(axes, list):
            # a list a axes have been passed
            for ax in axes:
                log.debug('add axe: {}'.format(ax.name))
                self._axes[ax.name] = ax
        elif isinstance(axes, dict):
            self._axes.update(axes)
        elif isinstance(axes, Axes):
            # it's an axe! add it to our list
            self._axes[axes.name] = axes

    @property
    def ax(self):
        """
        the main matplotlib axe associated to this dataset

        """
        return self._axes['main']

    @property
    def axec(self):
        """
        Matplotlib colorbar axe associated to this dataset

        """
        return self._axes['colorbar']

    @property
    def axex(self):
        """
        Matplotlib projection x axe associated to this dataset

        """
        return self._axes['xproj']

    @property
    def axey(self):
        """
        Matplotlib projection y axe associated to this dataset

        """
        return self._axes['yproj']

    @property
    def divider(self):
        """
        Matplotlib plot divider

        """
        return self._divider

    # -------------------------------------------------------------------------
    # events
    # -------------------------------------------------------------------------

    @classmethod
    def _onclick(cls, event):
        # not implemented here but in subclass
        pass


def curfig(hold=False, figsize=None):
    """
    Get the figure where to plot.

    Parameters
    ----------

    hold : `bool`, optioanl, False by default

        If hold is True, the plot will be issued on the last drawn figure

    figsize : `tuple`, optional

        A tuple representing the size of the figure in inch

    Returns
    -------

    fig : the figure object on which following plotting command will be issued

    """
    n = plt.get_fignums()

    if not n or not hold:
        # create a figure
        return plt.figure(figsize=figsize)

    # a figure already exists - if several we take the last
    return plt.figure(n[-1])

def curfig(hold=False, figsize=None):
    """
    Get the figure where to plot.

    Parameters
    ----------

    hold : `bool`, optioanl, False by default

        If hold is True, the plot will be issued on the last drawn figure

    figsize : `tuple`, optional

        A tuple representing the size of the figure in inch

    Returns
    -------

    fig : the figure object on which following plotting command will be issued

    """
    n = plt.get_fignums()

    if not n or not hold:
        # create a figure
        return plt.figure(figsize=figsize)

    # a figure already exists - if several we take the last
    return plt.figure(n[-1])


def show():
    """
    Method to force the `matplotlib` figure display

    """
    if not plotoptions.do_not_block or plt.isinteractive():
        if curfig(True):  # True to avoid opening a new one
            plt.show()

def set_figure_style(**kwargs):

    # set temporarity a new style if any
    # ----------------------------------
    plt.style.use('classic')
    plt.style.use(plotoptions.style)
    style = kwargs.get('style', None)

    if style:
        if not is_sequence(style):
            style = [style]
        if isinstance(style, dict):
            style = [style]
        style = [plotoptions.style] + list(style)
        plt.style.use(style)

    # other properties
    # ---------------------------------------

    fontsize = mpl.rcParams['font.size'] = \
        kwargs.get('fontsize', mpl.rcParams['font.size'])
    mpl.rcParams['legend.fontsize'] = int(fontsize * .8)
    mpl.rcParams['xtick.labelsize'] = int(fontsize)
    mpl.rcParams['ytick.labelsize'] = int(fontsize)


def available_styles():
    return ['notebook', 'paper', 'poster', 'talk', 'sans']


plot = NDIO.plot
load = NDIO.load
read = NDIO.read
write = NDIO.write

if __name__ == '__main__':

    # test interactive masks

    from spectrochempy.api import *

    source = NDDataset.read_omnic(
    os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))

    def _test_interactive_masks():

        source.interactive_masks(kind='stack', colorbar=True, figsize=(9,4))

    def _test_save():

        source.save('essai')

        source.plot_stack()
        source.save('essai2')

        os.remove(os.path.join(scpdata, 'essai.scp'))
        os.remove(os.path.join(scpdata, 'essai2.scp'))

    _test_save()


