# -*- coding: utf-8 -*-
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

import matplotlib

# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QDialog, QSizePolicy, QVBoxLayout,
                             QDialogButtonBox)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Python and third parties imports
# ----------------------------------

import copy

from matplotlib.widgets import SpanSelector
from matplotlib.lines import Line2D

from traitlets import (List, Bool, HasTraits, Instance, observe)

# local import
# ------------
from spectrochempy.core.plotters.multiplot import plot_with_transposed
from spectrochempy.core.dataset.ndarray import masked
from spectrochempy.core.plotters.utils import cmyk2rgb
from spectrochempy.application import app

plotoptions = app.plotoptions
log = app.log
options = app

# For color blind people, it is safe to use only 4 colors in graphs:
# see http://jfly.iam.u-tokyo.ac.jp/color/ichihara_etal_2008.pdf
#   Black CMYK=0,0,0,0
#   Red CMYK= 0, 77, 100, 0 %
#   Blue CMYK= 100, 30, 0, 0 %
#   Green CMYK= 85, 0, 60, 10 %
NBlack = (0, 0, 0)
NRed = cmyk2rgb(0, 77, 100, 0)
NBlue = cmyk2rgb(100, 30, 0, 0)
NGreen = cmyk2rgb(85, 0, 60, 10)

__all__ = []


class PlotWidget(FigureCanvas) :
    def __init__(self, parent, **kwargs) :
        figsize = kwargs.pop('figsize', (12, 8))

        self.fig = fig = Figure(figsize=figsize, dpi=100)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class MaskDialogGui(QDialog) :

    def __init__(self, dataset, **kwargs) :

        super(MaskDialogGui, self).__init__()

        # main layout
        layout = QVBoxLayout(self)
        self.setWindowTitle("Interactive Mask Selection")

        # add the plot widget where the plot will be done
        mw = self.mw = PlotWidget(self, **kwargs)
        layout.addWidget(mw)

        #toolbar = NavigationToolbar(mw, self)
        #layout.addWidget(toolbar)

        # Apply and Cancel buttons
        buttons = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal,
                self)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Set the controller
        self.controller = MaskController(dataset, fig=self.mw.fig, **kwargs)

        self.show()

    # .......................................................................
    def connect_all_events(self) :
        # self._button_press_event = self.mw.fig.canvas.mpl_connect(
        #        'button_press_event', self.controller._onclick)

        self._motion_notify_event = self.mw.fig.canvas.mpl_connect(
                'motion_notify_event', self.controller._onmove)

        # self._axes_leave_event = self.mw.fig.canvas.mpl_connect(
        #        'axes_leave_event', self.controller._onleaveaxes)

        self._key_press_event = self.mw.fig.canvas.mpl_connect(
                'key_press_event', self.controller._on_key)

        self._key_release_event = self.mw.fig.canvas.mpl_connect(
                'key_release_event', self.controller._on_key_release)

        # self._pick_event = self.mw.fig.canvas.mpl_connect('pick_event',
        #        self.controller._onpick)

    def start(self) :
        done = self.controller.start()

        self.connect_all_events()

        self.exec_()

class MaskController(HasTraits) :
    # a flag to say if we act on zoom
    _zoom_detection = Bool

    _all_masks = List()  # to store temporary the mask positions

    _xlim = List()  # used to detect zoom in ax
    _ylim = List()  # used to detect zoom in transposed axT

    _hover_T_line = Instance(Line2D, allow_none=True)
    _hover_line = Instance(Line2D, allow_none=True)

    # .........................................................................
    def __init__(self, source, fig=None, **kwargs) :
        """
        Parameters
        ----------
        source

        colorbar

        kwargs


        """
        # TODO: make it for 1D too!

        self.source = source
        #
        colorbar = kwargs.get('colorbar', True)

        # simplify plot for better interactivity
        plt.rcParams['path.simplify_threshold'] = 0

        # reduce the number of lines (max = max_lines_in_stack per default)
        self._maxlines = kwargs.get('maxlines', plotoptions.max_lines_in_stack)

        # draw initial plot with the transposed
        plot_with_transposed(source=source, fig=fig, colorbar=colorbar,
                suptitle_color=NBlue, **kwargs)

        self.fig = fig = source.fig
        self.ax = ax = source.ax
        self.axT = axT = source.axT
        self.axec = axec = source.axec
        self.axecT = axecT = source.axecT

        # set the working set of lines (a copy to preserve original list)
        # ax_lines and axT_lines are the displayed set that can be modified
        # while _ax_lines and _axT_lines are the original

        self.ax_lines = source._ax_lines[:]
        self.axT_lines = source._axT_lines[:]
        self.ax_zorder = len(source._ax_lines) + 2
        self.axT_zorder = len(source._axT_lines) + 2

    def get_limits(self) :
        # get limits (if they change, they will triger a change observed
        # below in the self._limits_changed function
        self._xlim = self.ax.get_xlim()
        self._ylim = self.axT.get_xlim()

    def exact_coord_x(self, c) :
        # get the closest nddataset x coordinate from c
        idx = self.source._loc2index(c, -1)
        return (idx, self.source.x.data[idx])

    def exact_coord_y(self, c) :
        # get the closest nddataset y coordinate from c
        idx = self.source._loc2index(c, 0)
        return (idx, self.source.y.data[idx])

    def get_hover_T_line(self, c) :
        idx, _ = self.exact_coord_x(c)
        self._hover_T_line = copy.copy(self.source._axT_lines[idx])
        self._hover_T_line.set_linewidth(2)
        self._hover_T_line.set_color(NGreen)
        self._hover_T_line.set_zorder(self.axT_zorder)
        #self._update_axes()

    def get_hover_line(self, c) :
        idx, _ = self.exact_coord_y(c)
        self._hover_line = copy.copy(self.source._ax_lines[idx])
        self._hover_line.set_linewidth(2)
        self._hover_line.set_color(NRed)
        self._hover_line.set_zorder(self.ax_zorder)
        #self._update_axes()

    def start(self) :

        # hide messages
        # -------------
        self.hide_help()
        self.hide_action()

        # utility transform for drawinfg on the colorbars
        # ------------------------------------------------

        ax_axis_to_data = self.ax.transAxes + self.ax.transData.inverted()
        ax_data_to_axis = ax_axis_to_data.inverted()
        self.axdata2axis = lambda x : ax_data_to_axis.transform((x, 0))[0]
        axT_axis_to_data = self.axT.transAxes + self.axT.transData.inverted()
        axT_data_to_axis = axT_axis_to_data.inverted()
        self.axTdata2axis = lambda x : axT_data_to_axis.transform((x, 0))[0]

        # cursors
        # ---------

        # init them out of the axes (not visible)
        self.x0 = min(self.ax.get_xlim()) - 10.
        self.y0 = min(self.axT.get_xlim()) - 10.

        self.ax_cursor = self.ax.axvline(self.x0, alpha=.8, color=NRed, lw=1.5,
                                         label='cursor', zorder=self.ax_zorder)
        self.axecT_cursor = self.axecT.axhline(self.axdata2axis(self.x0),
                                               alpha=.8, color=NGreen, lw=1.5,
                                               label='cursor')

        self.axT_cursor = self.axT.axvline(self.y0, alpha=.8, color=NGreen,
                                           lw=1.5, label='cursor',
                                           zorder=self.axT_zorder)
        self.axec_cursor = self.axec.axhline(self.axTdata2axis(self.y0),
                                             alpha=.8, color=NRed, lw=1.5,
                                             label='cursor')

        # row or cols to be masked initialisation
        # ---------------------------------------

        # self._all_masks will contain full informations about
        # the selected masks
        self._all_masks = []

        # update axes display data with a reduce number of lines
        # -------------------------------------------------------
        self._donotupdate = False
        self._update_axes()

        # now we can start to watch for changes (zoom, etc...)
        # ----------------------------------------------------
        self._zoom_detection = True

    def spans(self) :

        def _onspanx(xmin, xmax) :
            xmin, xmax = sorted((xmin, xmax))
            sp = self.ax.axvspan(xmin, xmax, facecolor='white',
                                 edgecolor='white', zorder=self.ax_zorder,
                                 picker=True)
            self._all_masks.append(('colspan', sp, xmin, xmax))
            self.show_action(
                "col span between {:.2f} and {:.2f}".format(xmin, xmax))

            self._donotupdate = False
            self.connections = self.connect_all_events()

            self._update_axes()

        def _onmovespan(xmin, xmax) :
            for conn in self.connections :
                try :
                    pass  # self.fig.canvas.mpl_disconnect(conn)
                except :
                    pass

                    # self._donotupdate = True

        min_x_span = np.diff(self.source.x.data)[0] * .1
        self._spanx = SpanSelector(self.ax, _onspanx, 'horizontal',
                                   minspan=min_x_span, button=[1],
                                   onmove_callback=_onmovespan, useblit=False,
                                   rectprops=dict(alpha=0.5,
                                                  zorder=self.axT_zorder,
                                                  facecolor=NRed,
                                                  edgecolor='w'))

        def _onspany(ymin, ymax) :
            ymin, ymax = sorted((ymin, ymax))
            sp = self.axT.axvspan(ymin, ymax, facecolor='white',
                                  edgecolor='white', zorder=self.axT_zorder,
                                  picker=True)
            self._all_masks.append(('rowspan', sp, ymin, ymax))
            self.show_action(
                "row span between {:.2f} and {:.2f}".format(ymin, ymax))

            self._donotupdate = False
            self.connections = self.connect_all_events()

            self._update_axes()

        min_y_span = np.diff(self.source.y.data)[0] * .1
        self._spany = SpanSelector(self.axT, _onspany, 'horizontal',
                                   minspan=min_y_span, button=[1],
                                   onmove_callback=_onmovespan, useblit=False,
                                   rectprops=dict(alpha=0.5,
                                                  zorder=self.axT_zorder,
                                                  facecolor=NGreen,
                                                  edgecolor='w'))

    # mouse events
    # ------------

    def _onleaveaxes(self, event) :
        # fired on a mouse motion leading to leave an ax
        # here we delete displayed cursor on all axes

        self.axT_cursor.set_xdata(self.y0)
        self.axec_cursor.set_ydata(self.axTdata2axis(self.y0))

        self.ax_cursor.set_xdata(self.x0)
        self.axecT_cursor.set_ydata(self.axdata2axis(self.y0))

        self._hover_line = None
        self._hover_T_line = None

        self._update_axes()

    def _onmove(self, event) :
        # fired on a mouse motion
        # we use this event to display cursors
        if not event.inaxes :
            return

        # self.hide_help()
        # self.hide_action()
        # and to get the new limts in case for example
        # of an interative zoom
        self.get_limits()

        if event.inaxes is self.ax :

            # ax  cursor at the x values
            x = event.xdata
            self.ax_cursor.set_xdata(x)

            # color bar marker at the value yT
            yT = self.axdata2axis(x)
            self.axecT_cursor.set_ydata(yT)

            # show the line in the transposed window
            self.get_hover_T_line(x)

        elif event.inaxes is self.axT :
            x = event.xdata
            self.axT_cursor.set_xdata(x)
            self.axec_cursor.set_ydata(self.axTdata2axis(x))
            self.get_hover_line(x)

        self._update_axes()


    def _onclick(self, event) :
        # fired on a mouse click.

        # if it is not fired in ax or the transposed axT, return
        # immediately and do nothing, except ot hide the 'help' text.
        self.hide_help()

        if event.inaxes not in [self.ax, self.axT] :
            return

        # check which button was pressed

        if event.button == 1 and event.dblclick :  # double-click left button
            inax = event.inaxes
            x = event.xdata

            if inax is self.axT :
                # set x to the closest original nddataset y coordinate
                idx, x = self.exact_coord_y(x)
                axvT = inax.axvline(x, lw=2, color='white', alpha=.9,
                                    picker=True, zorder=self.axT_zorder + 10,
                                    label='mask_row_%d' % idx)
                self._all_masks.append(('row', axvT, x, idx))

                # corresponding value in the original display
                # it is a complete row that we remove
                line = self.source._ax_lines[idx]
                self.show_action('mask row at y={:.2f}'.format(x))
                self._update_axes()

            elif inax is self.ax :
                idx, x = self.exact_coord_x(x)
                axv = inax.axvline(x, lw=2, color='white', alpha=.9,
                                   picker=True, zorder=self.ax_zorder + 10,
                                   label='mask_col_%d' % idx)
                self._all_masks.append(('col', axv, x, idx))

                # corresponding value in the transposed display
                # it is a complete row of axT

                # corresponding value in the original display
                # it is a complete row that we remove
                line = self.source._axT_lines[idx]
                self.show_action('mask column at x={:.2f}'.format(x))
                self._update_axes()

    # key events
    # ----------

    def _on_key(self, event) :

        if event.key in ['h', 'a'] :
            # we show the help.
            self.show_help()

        if event.key in ['esc'] :
            # we show the help.
            self.hide_help()
            self.hide_action()
            self.self.fig.canvas.draw()

    def _on_key_release(self, event) :

        if event.key in ['a', 'h'] :
            self.hide_help()

        if event.key in ['ctrl+z'] :
            if self._all_masks :
                last = list(self._all_masks.pop(-1))
                # if last[0] in ['rowspan', 'colspan', '''col']:
                last[1].remove()
                # else:
                #    last[1].set_color(last[3])
                #    last[1].set_linewidth(last[4])
                self.show_action(
                        'removed {} selection at {:.2f}'.format(last[0],
                                last[2]))
                self.self.fig.canvas.draw()
            else :
                self.show_action('all masks were already removed')

        if event.key in ['ctrl+x'] :
            log.info("apply all defined mask to the dataset")

            for item in self._all_masks :
                _item = list(item)
                if _item[0] in ['colspan'] :
                    xmin, xmax = _item[2 :]
                    self[:, xmin :xmax] = masked
                    #  log.debug("span {}:{}".format(xmin, xmax))

                elif _item[0] in ['rowspan'] :
                    ymin, ymax = _item[2 :]
                    self[ymin :ymax] = masked
                    #  log.debug("rowspan {}:{}".format(ymin, ymax))

                elif _item[0] in ['col'] :
                    x = _item[2]
                    self[:, x] = masked
                    #  log.debug("col {}".format(x))

                elif _item[0] in ['row'] :
                    y = eval(_item[2])
                    self[y] = masked
                    #  log.debug("row {}".format(y))

            self.show_action('Masks applied to the dataset')

            plt.close(self._fig)

    # pick event
    # ----------

    def _onpick(self, event) :

        inax = event.mouseevent.inaxes

        if isinstance(event.artist, Line2D) :

            button = event.mouseevent.button
            sel = event.artist

            if sel.get_label().startswith('mask') :
                # TODO: offer the possibility of deleting this mask selection
                return

            if inax is self.ax :
                y = eval(sel.get_label())
                x = event.mouseevent.xdata
                cond_row = (button == 3)
                cond_col = (button == 1 and event.mouseevent.dblclick)

            elif inax is self.axT :
                x = eval(sel.get_label())
                y = event.mouseevent.xdata
                cond_col = (button == 3)
                cond_row = (button == 1 and event.mouseevent.dblclick)

            if cond_row :
                # right button -> row selection
                idx, y = self.exact_coord_y(y)
                axvT = self.axT.axvline(y, lw=2, color='white', alpha=.9,
                                        picker=True,
                                        zorder=self.axT_zorder + 10,
                                        label='mask_row_%d' % idx)
                self._all_masks.append(('row', axvT, y, idx))
                self.show_action("mask row picked at y={:.2f}".format(y))

            elif cond_col :

                # left button -> column selection
                idx, x = self.exact_coord_x(x)
                axv = self.ax.axvline(x, lw=2, color='white', alpha=.9,
                                      picker=True, zorder=self.ax_zorder + 10,
                                      label='mask_col_%d' % idx)
                self._all_masks.append(('col', axv, x, idx))
                self.show_action("mask column picked at x={:.2f}".format(x))

            self._update_axes()

    # .........................................................................
    def _get_masked_lines(self) :

        # get the masks related lines and markers
        masked_lines = {}
        masked_T_lines = {}
        masked_markers = []
        masked_T_markers = []

        for item in self._all_masks :

            _item = list(item)

            if _item[0].endswith('span') :

                direction = _item[0][:3]
                val1, val2 = sorted(_item[2 :])
                if direction == 'row' :
                    lines = self.ax_lines[:]
                    axlines = self.ax_lines
                elif direction == 'col' :
                    lines = self.axT_lines[:]
                    axlines = self.axT_lines
                for line in lines :
                    select = (eval(line.get_label()) >= val1 and eval(
                        line.get_label()) <= val2)
                    if select and line in axlines :
                        axlines.remove(line)

            elif _item[0] == 'col' :

                idx = _item[3]
                line = self.source._axT_lines[idx]
                masked_T_lines[line.get_label()] = line

                axv = _item[1]
                masked_markers.append(axv)

            elif _item[0] == 'row' :

                idx = _item[3]
                line = self.source._ax_lines[idx]
                masked_lines[line.get_label()] = line

                axv = _item[1]
                masked_T_markers.append(axv)

        return (masked_lines, masked_T_lines, masked_markers, masked_T_markers)

    # .........................................................................
    def _update_axes(self) :
        #
        if self._donotupdate :
            return

        (masked_lines, masked_T_lines, masked_markers,
         masked_T_markers) = self._get_masked_lines()

        self.ax.lines = []
        setpy = max(len(self.ax_lines) // self._maxlines, 1)
        self.ax.lines = self.ax_lines[::setpy]  # displayed ax lines

        for line in masked_lines.values() :  # remove masked line
            if line in self.ax.lines :
                self.ax.lines.remove(line)

        self.ax.lines.append(self.ax_cursor)  # restore cursor line

        if self._hover_line is not None :  # \
            # and self._hover_line.get_label() not in masked_lines.keys(
            # ):  # show hover line
            #  log.debug(self._hover_line.get_label())
            self.ax.lines.append(self._hover_line)

        for line in masked_markers :
            self.ax.lines.append(line)

        self.axT.lines = []
        setpx = max(len(self.axT_lines) // self._maxlines, 1)
        self.axT.lines = self.axT_lines[::setpx]  # displayed axTlines

        for line in masked_T_lines.values() :
            if line in self.axT.lines :
                self.axT.lines.remove(line)

        self.axT.lines.append(self.axT_cursor)  # restore cursor line

        if self._hover_T_line is not None :  # \
            # and self._hover_T_line.get_label() not in masked_T_lines.keys():
            self.axT.lines.append(self._hover_T_line)

        for line in masked_T_markers :
            self.axT.lines.append(line)

        xx = [line.get_label() for line in self.axT.lines[-5 :]]
        print("\n\n", xx)

        self.fig.canvas.draw()

    # .........................................................................
    @observe('_xlim', '_ylim')
    def _limits_changed(self, change) :
        if not self._zoom_detection :
            return
        self.ax_lines = self.source._ax_lines[:]
        self.axT_lines = self.source._axT_lines[:]

        if change['name'] == '_xlim' :
            x1, x2 = sorted(self._xlim)
            lines = self.axT_lines[:]
            for line in lines :
                if eval(line.get_label()) < x1 or eval(line.get_label()) > x2 :
                    self.axT_lines.remove(line)

        if change['name'] == '_ylim' :
            y1, y2 = sorted(self._ylim)
            lines = self.ax_lines[:]
            for line in lines :
                if eval(line.get_label()) < y1 or eval(line.get_label()) > y2 :
                    self.ax_lines.remove(line)
        self._update_axes()





if __name__ == '__main__' :

    from spectrochempy.api import *

    options.log_level = DEBUG

    A = NDDataset.read_omnic(
            os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))

    # some preprocessing
    A.y -= A.y[0]
    A.y.to('hour', inplace=True)
    A.y.title = u'Aquisition time'

    from PyQt5.QtWidgets import QApplication

    #: Handler to the GUI underlying application
    guiApp = QApplication(sys.argv)
    w = MaskDialogGui(A)
    print(w)

    sys.exit(guiApp.exec_())
