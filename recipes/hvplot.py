# -*- coding: utf-8 -*-

# BSD 3-Clause License

# Copyright (c) 2018, PyViz
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__all__ = ['hvplot']
__dataset_methods__ = __all__

# This module is an adaptation of hvplot to spectrochempy
# Setup of holoviews
import os
import warnings
from contextlib import redirect_stdout
from IPython import get_ipython
from spectrochempy.api import IN_IPYTHON, kernel, NO_DISPLAY
from .scpy import SCPYArrayInterface
import holoviews as hv
from holoviews.core.data.interface import Interface

if IN_IPYTHON and kernel and not NO_DISPLAY:
    # hack to prevent display of three empty lines from holoview in the notebook
    ip = get_ipython()
    ip.run_cell_magic("capture", line='',
                      cell= \
                          """
                          import holoviews as hv
                          hv.extension('bokeh', 'matplotlib', logo=False)
                          """
                      )

import difflib
import pandas as pd
import numpy as np

from traitlets import (HasTraits, Dict, Any, Instance, Unicode, This, default)

from bokeh.models import HoverTool
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import DynamicMap, HoloMap, Callable
from holoviews.core.overlay import NdOverlay
from holoviews.core.options import Store, Cycle
from holoviews.core.layout import NdLayout
from holoviews.core.util import max_range
from holoviews.element import (
    Curve, Scatter, Area, Bars, BoxWhisker, Dataset, Distribution,
    Table, HeatMap, Image, HexTiles, QuadMesh, Bivariate, Histogram,
    Violin, Contours, Polygons, Points, Path, Labels, RGB
)

from holoviews.plotting.util import process_cmap
from holoviews.operation import histogram
from holoviews.streams import Buffer, Pipe
from holoviews.util.transform import dim

from bokeh.io import export_png as _export_png, show as _show, save as _save
from bokeh.resources import CDN as _CDN
from holoviews import Store

from spectrochempy.core.dataset.nddataset import NDDataset

renderer = hv.renderer('bokeh')


@property
def hvplot(dataset):
    return hvPlot(dataset)


def check_library(obj, library):
    if not isinstance(library, list):
        library = [library]
    return any([obj.__module__.split('.')[0].startswith(l) for l in library])


def is_series(data):
    if not check_library(data, ['dask', 'pandas']):
        return False
    elif isinstance(data, pd.Series):
        return True
    elif check_library(data, 'dask'):
        import dask.dataframe as dd
        return isinstance(data, dd.Series)
    else:
        return False


def is_spectrochempy(data):
    if not check_library(data, 'spectrochempy'):
        return False
    return isinstance(data, (NDDataset))


class hvPlot(HasTraits):
    _data = Instance(NDDataset)
    _plots = Dict({})
    _metadata = Dict({})

    def __init__(self, data, custom_plots={}, **metadata):
        self._data = data
        self._plots = custom_plots
        self._metadata = metadata

    def __call__(self, x=None, y=None, kind=None, **kwds):
        """
        Default plot method (for more detailed options use specific
        plot method, e.g. df.hvplot.line)

        Parameters
        ----------
        x, y: string, optional
            Field name in the data to draw x- and y-positions from
        kind: string, optional
            The kind of plot to generate, e.g. 'line', 'scatter', etc.
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        HoloViews object: Object representing the requested visualization
        """
        return self._get_converter(x, y, kind, **kwds)(kind, x, y)

    def _get_converter(self, x=None, y=None, kind=None, **kwds):
        params = dict(self._metadata, **kwds)
        x = x or params.pop('x', None)
        y = y or params.pop('y', None)
        kind = kind or params.pop('kind', None)
        return HoloViewsConverter(
            self._data, x, y, kind=kind, **params
        )

    def __dir__(self):
        """
        List default attributes and custom defined plots.
        """
        dirs = super(hvPlot, self).__dir__()
        return sorted(list(dirs) + list(self._plots))

    def __getattribute__(self, name):
        """
        Custom getattribute to expose user defined subplots.
        """
        if name != '_plots':  # avoid to crash
            return super(hvPlot, self).__getattribute__(name)

        plots = object.__getattribute__(self, '_plots')
        # plots = self._plots

        if name in plots:
            plot_opts = plots[name]
            if 'kind' in plot_opts and name in HoloViewsConverter._kind_mapping:
                warnings.warn("Custom options for existing plot types should not "
                              "declare the 'kind' argument. The .%s plot method "
                              "was unexpectedly customized with kind=%r."
                              % (plot_opts['kind'], name))
                plot_opts['kind'] = name
            return hvPlot(self._data, **dict(self._metadata, **plot_opts))
        return super(hvPlot, self).__getattribute__(name)

    def line(self, x=None, y=None, **kwds):
        """
        Line plot

        Parameters
        ----------
        x, y: string, optional
            Field name to draw x- and y-positions from
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        HoloViews object: Object representing the requested visualization
        """
        return self(x, y, kind='line', **kwds)

    def step(self, x=None, y=None, where='mid', **kwds):
        """
        Step plot

        Parameters
        ----------
        x, y: string, optional
            Field name to draw x- and y-positions from
        where: string, optional
            Defines where the steps are placed (options: 'mid' (default), 'pre' and 'post')
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        HoloViews object: Object representing the requested visualization
        """
        return self(x, y, kind='step', **kwds)

    def scatter(self, x=None, y=None, **kwds):
        """
        Scatter plot

        Parameters
        ----------
        x, y: string, optional
            Field name to draw x- and y-positions from
        c: string, optional
            Name of the field to color points by
        s: string, optional
            Name of the field to scale point size by
        scale: number, optional
            Scaling factor to apply to point scaling
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, kind='scatter', **kwds)

    def area(self, x=None, y=None, y2=None, stacked=True, **kwds):
        """
        Area plot

        Parameters
        ----------
        x, y, y2: string, optional
            Field name to draw x- and y-positions from
        stacked: boolean
            Whether to stack multiple areas
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        if 'alpha' not in kwds and not stacked:
            kwds['alpha'] = 0.5
        return self(x, y, y2=y2, kind='area', stacked=stacked, **kwds)

    def heatmap(self, x=None, y=None, C=None, colorbar=True, **kwds):
        """
        HeatMap plot

        Parameters
        ----------
        x, y: string, optional
            Field name to draw x- and y-positions from
        C: string
            Field to draw heatmap color from
        colorbar: boolean
            Whether to display a colorbar
        reduce_function: function
            Function to compute statistics for heatmap
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, kind='heatmap', C=C, colorbar=colorbar, **kwds)

    def hexbin(self, x=None, y=None, C=None, colorbar=True, **kwds):
        """
        HexBin plot

        Parameters
        ----------
        x, y: string, optional
            Field name to draw x- and y-positions from
        C: string
            Field to draw heatmap color from
        colorbar: boolean
            Whether to display a colorbar
        reduce_function: function
            Function to compute statistics for hexbins
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, kind='hexbin', C=C, colorbar=colorbar, **kwds)

    def bivariate(self, x=None, y=None, colorbar=True, **kwds):
        """
        Bivariate plot for 2D density

        Parameters
        ----------
        x, y: string, optional
            Field name to draw x- and y-positions from
        colorbar: boolean
            Whether to display a colorbar
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, kind='bivariate', colorbar=colorbar, **kwds)

    def bar(self, x=None, y=None, **kwds):
        """
        Bars plot

        Parameters
        ----------
        x, y: string, optional
            Field name to draw x- and y-positions from
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, kind='bar', **kwds)

    def barh(self, x=None, y=None, **kwds):
        """
        Horizontal bar plot

        Parameters
        ----------
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, kind='barh', **kwds)

    def box(self, y=None, by=None, **kwds):
        """
        Boxplot

        Parameters
        ----------
        y: string or sequence
            Column in the DataFrame to compute distribution on.
        by: string or sequence
            Column in the DataFrame to group by.
        kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(kind='box', x=None, y=y, by=by, **dict(kwds, hover=False))

    def violin(self, y=None, by=None, **kwds):
        """
        Boxplot

        Parameters
        ----------
        y: string or sequence
            Column in the DataFrame to compute distribution on.
        by: string or sequence
            Column in the DataFrame to group by.
        kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(kind='violin', x=None, y=y, by=by, **dict(kwds, hover=False))

    def hist(self, y=None, by=None, **kwds):
        """
        Histogram

        Parameters
        ----------
        y: string or sequence
            Column in the DataFrame to compute distribution on.
        by: string or sequence
            Column in the DataFrame to group by.
        kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(kind='hist', x=None, y=y, by=by, **kwds)

    def kde(self, y=None, by=None, **kwds):
        """
        KDE

        Parameters
        ----------
        y: string or sequence
            Column in the DataFrame to compute distribution on.
        by: string or sequence
            Column in the DataFrame to group by.
        kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(kind='kde', x=None, y=y, by=by, **kwds)

    def table(self, columns=None, **kwds):
        """
        Table

        Parameters
        ----------
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(kind='table', **dict(kwds, columns=columns))

    def dataset(self, columns=None, **kwds):
        """
        Dataset

        Parameters
        ----------
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(kind='dataset', **dict(kwds, columns=columns))

    def image(self, x=None, y=None, z=None, colorbar=True, **kwds):
        """
        Image plot

        Parameters
        ----------
        x, y: string, optional
            The coordinate variable along the x- and y-axis
        z: string, optional
            The data variable to plot
        colorbar: boolean
            Whether to display a colorbar
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, z=z, kind='image', colorbar=colorbar, **kwds)

    def rgb(self, x=None, y=None, z=None, bands=None, **kwds):
        """
        RGB plot

        Parameters
        ----------
        x, y: string, optional
            The coordinate variable along the x- and y-axis
        bands: string, optional
            The coordinate variable to draw the RGB channels from
        z: string, optional
            The data variable to plot
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        if bands is not None:
            kwds['bands'] = bands
        return self(x, y, z=z, kind='rgb', **kwds)

    def quadmesh(self, x=None, y=None, z=None, colorbar=True, **kwds):
        """
        QuadMesh plot

        Parameters
        ----------
        x, y: string, optional
            The coordinate variable along the x- and y-axis
        z: string, optional
            The data variable to plot
        colorbar: boolean
            Whether to display a colorbar
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, z=z, kind='quadmesh', colorbar=colorbar, **kwds)

    def contour(self, x=None, y=None, z=None, colorbar=True, **kwds):
        """
        Line contour plot

        Parameters
        ----------
        x, y: string, optional
            The coordinate variable along the x- and y-axis
        z: string, optional
            The data variable to plot
        levels: int, optional
            The number of contour levels
        colorbar: boolean
            Whether to display a colorbar
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, z=z, kind='contour', colorbar=colorbar, **kwds)

    def contourf(self, x=None, y=None, z=None, colorbar=True, **kwds):
        """
        Filled contour plot

        Parameters
        ----------
        x, y: string, optional
            The coordinate variable along the x- and y-axis
        z: string, optional
            The data variable to plot
        levels: int, optional
            The number of contour levels
        colorbar: boolean
            Whether to display a colorbar
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, z=z, kind='contourf', colorbar=colorbar, **kwds)

    def points(self, x=None, y=None, **kwds):
        """
        Point plot use for 2D coordinate systems

        Parameters
        ----------
        x, y: string, optional
            The coordinate variable along the x- and y-axis
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, kind='points', **kwds)

    def labels(self, x=None, y=None, text=None, **kwds):
        """
        Labels plot.

        Parameters
        ----------
        x, y: string, optional
            The coordinate variable along the x- and y-axis
        text: string, optional
            The column to draw the text labels from
        **kwds: optional
            Keyword arguments to pass on to
            :py:meth:`hvplot.converter.HoloViewsConverter`.
        Returns
        -------
        obj: HoloViews object
            The HoloViews representation of the plot.
        """
        return self(x, y, text=text, kind='labels', **kwds)


def save(obj, filename, title=None, resources=None):
    """
    Saves HoloViews objects and bokeh plots to file.

    Parameters
    ----------
    obj: HoloViews object
       HoloViews object to export
    filename: string
       Filename to save the plot to
    title: string
       Optional title for the plot
    resources: bokeh resources
       One of the valid bokeh.resources (e.g. CDN or INLINE)
    """
    if isinstance(obj, hv.core.Dimensioned):
        plot = renderer.get_plot(obj).state
    else:
        raise ValueError('%s type object not recognized and cannot be saved.' %
                         type(obj).__name__)

    if filename.endswith('png'):
        _export_png(plot, filename=filename)
        return
    if not filename.endswith('.html'):
        filename = filename + '.html'

    if title is None:
        title = 'hvPlot Plot'
    if resources is None:
        resources = _CDN

    if obj.traverse(lambda x: x, [hv.HoloMap]):
        renderer.save(plot, filename)
    else:
        _save(plot, filename, title=title, resources=resources)


def showbokeh(obj, filename=None):
    """
    Displays HoloViews objects in and outside the notebook

    Parameters
    ----------
    obj: HoloViews object
       HoloViews object to export
    """
    if not isinstance(obj, hv.core.Dimensioned):
        raise ValueError('%s type object not recognized and cannot be shown.' %
                         type(obj).__name__)

    if obj.traverse(lambda x: x, [hv.HoloMap]):
        renderer.app(obj, show=True, new_window=True)
    else:
        from bokeh.io import output_file
        output_file(filename or 'default.html')
        _show(renderer.get_plot(obj).state)


class HoloViewsConverter(object):
    """
    Generic options
    ---------------
    colorbar (default=False): boolean
        Enables colorbar
    flip_xaxis/flip_yaxis: boolean
        Whether to flip the axis left to right or up and down respectively
    grid (default=False): boolean
        Whether to show a grid
    hover (default=True): boolean
        Whether to show hover tooltips
    hover_cols (default=[]): list
        Additional columns to add to the hover tool
    invert (default=False): boolean
        Swaps x- and y-axis
    legend (default=True): boolean or str
        Whether to show a legend, or a legend position
        ('top', 'bottom', 'left', 'right')
    logx/logy (default=False): boolean
        Enables logarithmic x- and y-axis respectively
    loglog (default=False): boolean
        Enables logarithmic x- and y-axis
    padding: number or tuple
        Fraction by which to increase auto-ranged extents to make
        datapoints more visible around borders. Supports tuples to
        specify different amount of padding for x- and y-axis and
        tuples of tuples to specify different amounts of padding for
        upper and lower bounds.
    rot: number
        Rotates the axis ticks along the x-axis by the specified
        number of degrees.
    shared_axes (default=False): boolean
        Whether to link axes between plots
    title (default=''): str
        Title for the plot
    tools (default=[]): list
        List of tool instances or strings (e.g. ['tap', box_select'])
    xformatter/yformatter (default=None): str or TickFormatter
        Formatter for the x-axis and y-axis (accepts printf formatter,
        e.g. '%.3f', and bokeh TickFormatter)
    xlabel/ylabel (default=None): str
        Axis labels for the x-axis and y-axis
    xlim/ylim (default=None): tuple
        Plot limits of the x- and y-axis
    xticks/yticks (default=None): int or list
        Ticks along x- and y-axis specified as an integer, list of
        ticks positions, or list of tuples of the tick positions and labels
    width (default=800)/height (default=300): int
        The width and height of the plot in pixels

    Datashader options
    ------------------
    aggregator (default=None):
        Aggregator to use when applying rasterize or datashade operation
        (valid options include 'mean', 'count', 'min', 'max' and more, and
        datashader reduction objects)
    dynamic (default=True):
        Whether to return a dynamic plot which sends updates on widget and
        zoom/pan events or whether all the data should be embedded
        (warning: for large groupby operations embedded data can become
        very large if dynamic=False)
    datashade (default=False):
        Whether to apply rasterization and shading using datashader
        library returning an RGB object
    dynspread (default=False):
        Allows plots generated with datashade=True to increase the point
        size to make sparse regions more visible
    rasterize (default=False):
        Whether to apply rasterization using the datashader library
        returning an aggregated Image
    xsampling/ysampling (default=None):
        Declares a minimum sampling density beyond.
    """

    _gridded_types = ['image', 'contour', 'contourf', 'quadmesh', 'rgb']

    _stats_types = ['hist', 'kde', 'violin', 'box']

    _data_options = ['x', 'y', 'kind', 'by', 'use_index', 'use_dask',
                     'dynamic', 'value_label', 'group_label',
                     'backlog', 'persist']

    _axis_options = ['width', 'height', 'shared_axes', 'grid', 'legend',
                     'rot', 'xlim', 'ylim', 'xticks', 'yticks', 'colorbar',
                     'invert', 'title', 'logx', 'logy', 'loglog', 'xaxis',
                     'yaxis', 'xformatter', 'yformatter', 'xlabel', 'ylabel',
                     'padding']

    _style_options = ['color', 'alpha', 'colormap', 'fontsize', 'c']

    _op_options = ['datashade', 'rasterize', 'x_sampling', 'y_sampling',
                   'aggregator']

    _kind_options = {
        'scatter': ['s', 'c', 'scale', 'logz'],
        'step': ['where'],
        'area': ['y2'],
        'hist': ['bins', 'bin_range', 'normed', 'cumulative'],
        'heatmap': ['C', 'reduce_function', 'logz'],
        'hexbin': ['C', 'reduce_function', 'gridsize', 'logz'],
        'dataset': ['columns'],
        'table': ['columns'],
        'image': ['z', 'logz'],
        'rgb': ['z', 'bands'],
        'quadmesh': ['z', 'logz'],
        'contour': ['z', 'levels', 'logz'],
        'contourf': ['z', 'levels', 'logz'],
        'points': ['s', 'marker', 'c', 'scale', 'logz'],
        'polygons': ['logz', 'c'],
        'labels': ['text', 'c', 's']
    }

    _kind_mapping = {
        'line': Curve, 'scatter': Scatter, 'heatmap': HeatMap,
        'bivariate': Bivariate, 'quadmesh': QuadMesh, 'hexbin': HexTiles,
        'image': Image, 'table': Table, 'hist': Histogram, 'dataset': Dataset,
        'kde': Distribution, 'area': Area, 'box': BoxWhisker, 'violin': Violin,
        'bar': Bars, 'barh': Bars, 'contour': Contours, 'contourf': Polygons,
        'points': Points, 'polygons': Polygons, 'paths': Path, 'step': Curve,
        'labels': Labels, 'rgb': RGB
    }

    _colorbar_types = ['image', 'hexbin', 'heatmap', 'quadmesh', 'bivariate',
                       'contour', 'contourf', 'polygons']

    _legend_positions = ("top_right", "top_left", "bottom_left",
                         "bottom_right", "right", "left", "top",
                         "bottom")

    def __init__(self, data, x, y, kind=None, by=None, use_index=True,
                 group_label='Variable', value_label='value',
                 backlog=1000, persist=False, use_dask=False,
                 fields={}, groupby=None, dynamic=True,
                 width=700, height=300, shared_axes=True,
                 grid=False, legend=True, rot=None, title=None,
                 xlim=None, ylim=None, clim=None, xticks=None, yticks=None,
                 logx=False, logy=False, loglog=False, hover=True,
                 subplots=False, label=None, invert=False,
                 stacked=False, colorbar=None, fontsize=None,
                 colormap=None, datashade=False, rasterize=False,
                 row=None, col=None, figsize=None, debug=False,
                 xaxis=True, yaxis=True, framewise=True, aggregator=None,
                 precompute=False, flip_xaxis=False, flip_yaxis=False,
                 dynspread=False, hover_cols=[], x_sampling=None,
                 y_sampling=None, xlabel=None, ylabel=None,
                 xformatter=None, yformatter=None, tools=[], padding=None,
                 **kwds):

        # Process data and related options
        self._process_data(kind, data, x, y, by, groupby, row, col,
                           use_dask, persist, backlog, label, value_label,
                           hover_cols, kwds)
        self.use_index = use_index
        self.value_label = value_label
        self.group_label = group_label
        self.dynamic = dynamic
        self.row = row
        self.col = col

        # Operations
        self.datashade = datashade
        self.rasterize = rasterize
        self.dynspread = dynspread
        self.aggregator = aggregator
        self.precompute = precompute
        self.x_sampling = x_sampling
        self.y_sampling = y_sampling

        # By type
        self.subplots = subplots
        self._by_type = NdLayout if subplots else NdOverlay

        # Process options
        self.stacked = stacked
        style_opts, plot_opts, kwds = self._process_style(colormap, kwds)
        self.invert = invert
        plot_opts['logx'] = logx or loglog
        plot_opts['logy'] = logy or loglog
        plot_opts['show_grid'] = grid
        plot_opts['shared_axes'] = shared_axes
        plot_opts['show_legend'] = bool(legend)

        if legend in self._legend_positions:
            plot_opts['legend_position'] = legend
        elif legend in (True, False, None):
            plot_opts['legend_position'] = 'right'
        else:
            raise ValueError('The legend option should be a boolean or '
                             'a valid legend position (i.e. one of %s).'
                             % list(self._legend_positions))
        if xticks:
            plot_opts['xticks'] = xticks
        if yticks:
            plot_opts['yticks'] = yticks
        if not xaxis:
            plot_opts['xaxis'] = None
        if not yaxis:
            plot_opts['yaxis'] = None
        if xlabel is not None:
            plot_opts['xlabel'] = xlabel
        if ylabel is not None:
            plot_opts['ylabel'] = ylabel
        if xlim is not None:
            plot_opts['xlim'] = xlim
        if ylim is not None:
            plot_opts['ylim'] = ylim
        if padding is not None:
            plot_opts['padding'] = padding
        if xformatter is not None:
            plot_opts['xformatter'] = xformatter
        if yformatter is not None:
            plot_opts['yformatter'] = yformatter
        if flip_xaxis:
            plot_opts['invert_xaxis'] = True
        if flip_yaxis:
            plot_opts['invert_yaxis'] = True
        if width:
            plot_opts['width'] = width
        if height:
            plot_opts['height'] = height
        if fontsize:
            plot_opts['fontsize'] = fontsize
        if isinstance(colorbar, bool):
            plot_opts['colorbar'] = colorbar
        elif self.kind in self._colorbar_types:
            plot_opts['colorbar'] = True
        if invert:
            plot_opts['invert_axes'] = kind != 'barh'
        if rot:
            axis = 'yrotation' if invert else 'xrotation'
            plot_opts[axis] = rot

        tools = list(tools)
        if hover and not any(t for t in tools if isinstance(t, HoverTool)
                                                 or t == 'hover'):
            tools.append('hover')
        plot_opts['tools'] = tools

        if title is not None:
            plot_opts['title_format'] = title
        self._plot_opts = plot_opts
        options = Store.options(backend='bokeh')
        el_type = self._kind_mapping[self.kind].__name__
        style = options[el_type].groups['style']
        cycled_opts = [k for k, v in style.kwargs.items() if isinstance(v, Cycle)]
        for opt in cycled_opts:
            color = style_opts.get('color', None)
            if color is None:
                color = process_cmap(colormap or 'Category10', categorical=True)
            style_opts[opt] = Cycle(values=color) if isinstance(color, list) else color
        self._style_opts = style_opts
        self._norm_opts = {'framewise': framewise, 'axiswise': not shared_axes}
        self.kwds = kwds

        # Process dimensions and labels
        self.label = label
        self._relabel = {'label': label} if label else {}
        self._dim_ranges = {'c': clim or (None, None)}
        self._redim = fields

        # High-level options
        self._validate_kwds(kwds)
        if debug:
            kwds = dict(x=self.x, y=self.y, by=self.by, kind=self.kind,
                        groupby=self.groupby)
            warnings.warn('Plotting {kind} plot with parameters x: {x}, '
                          'y: {y}, by: {by}, groupby: {groupby}'.format(**kwds))

    @staticmethod
    def _process_spectrochempy(data, x, y, by, groupby, use_dask, persist, gridded, label, value_label, other_dims):

        org_data = data
        data_vars = [data.title] if isinstance(data, NDDataset) else list(data.data_vars)
        ignore = (by or []) + (groupby or [])

        dims = [c.title for (i, c) in enumerate(data.coords) if data.shape[i] > 1 and c.title not in ignore][::-1]
        # index_dims = [d for d in dims if d in dataset.indexes]

        if gridded:
            if len(dims) < 2:
                dims += [dim for dim in list(data.dims)[::-1] if dim not in dims]
            if not (x or y):
                x, y = data_vars[:2] if len(data_vars) > 1 else dims[:2]
            elif x and not y:
                y = [d for d in dims if d != x][0]
            elif y and not x:
                x = [d for d in dims if d != y][0]
            if len(dims) > 2 and not groupby:
                dims = list(data.coords[x].dims) + list(data.coords[y].dims)
                groupby = [d for d in data_vars if d not in (x, y) and d not in dims and d not in other_dims]
        else:
            if use_dask:  # TODO: use dask
                data = data.to_dask_dataframe()
                data = data.persist() if persist else data
            else:
                data = data.to_dataframe()
                if len(data.index.names) > 1:
                    data = data.reset_index()
            if len(dims) == 0:
                dims = ['index']
            if x and not y:
                y = dims[0] if x in data_vars else data_vars  #
            elif y and not x:
                x = data_vars[0] if y in dims else dims[0]
            elif not x and not y:
                x, y = dims[0], data_vars

            if by is None:
                by = [c for c in dims if c not in (x, y)]
                if len(by) > 1: by = []
            if groupby is None:
                groupby = [c for c in dims if c not in by + [x, y]]
        return data, x, y, by, groupby

    def _process_data(self, kind, data, x, y, by, groupby, row, col,
                      use_dask, persist, backlog, label, value_label,
                      hover_cols, kwds):
        gridded = kind in self._gridded_types
        gridded_data = False

        # Validate DataSource
        self.data_source = data
        self.is_series = is_series(data)

        if groupby is not None and not isinstance(groupby, list):
            groupby = [groupby]
        if by is not None and not isinstance(by, list):
            by = [by]

        if is_spectrochempy(data):

            ignore = (groupby or []) + (by or [])
            dims = [c.title for c in data.coords if c.shape != ()
                    and c not in ignore]

            if kind is None and (not (x or y) or all(c in data.coords for c in (x, y))):
                if len(dims) == 1:
                    kind = 'line'
                elif len(dims) == 2 or (x and y):
                    kind = 'image'
                    gridded = True
                else:
                    kind = 'hist'  # TODO: when this happens?

            datatype = 'dask' if use_dask else 'pandas'  # TODO: Dask is not implemented
            if gridded:
                datatype = 'spectrochempy'
                gridded_data = True

            other_dims = []
            data, x, y, by_new, groupby_new = self._process_spectrochempy(data, x, y, by, groupby,
                                                                          use_dask, persist, gridded,
                                                                          label, value_label, other_dims)

            if kind not in self._stats_types:
                if by is None: by = by_new
                if groupby is None: groupby = groupby_new

            if groupby:
                groupby = [g for g in groupby if g not in (row, col)]
            self.data = data
        else:
            raise ValueError('Supplied data type %s not understood' % type(data).__name__)

        # Validate data and arguments
        if by is None: by = []
        if groupby is None: groupby = []

        if gridded:
            if not gridded_data:
                raise ValueError('%s plot type requires gridded data, '
                                 'e.g. a NumPy array or xarray Dataset, '
                                 'found %s type' % (kind, type(self.data).__name__))
            not_found = [g for g in groupby if g not in data.coords]
            data_vars = [data.title] if isinstance(data, NDDataset) else list(data.data_vars)
            indexes = data.coordtitles
            self.variables = indexes + data_vars

            if groupby and not_found:
                raise ValueError('The supplied groupby dimension(s) %s '
                                 'could not be found, expected one or '
                                 'more of: %s' % (not_found, list(data.coords)))
        else:
            # Determine valid indexes
            if isinstance(self.data, pd.DataFrame):
                if self.data.index.names == [None]:
                    indexes = [self.data.index.name or 'index']
                else:
                    indexes = list(self.data.index.names)
            else:
                indexes = [c for c in self.data.reset_index().columns
                           if c not in self.data.columns]

            if len(indexes) == 2 and not (x or y or by):
                if kind == 'heatmap':
                    x, y = indexes
                elif kind in ('bar', 'barh'):
                    x, by = indexes

            # Rename non-string columns
            renamed = {c: str(c) for c in data.columns if not isinstance(c, hv.util.basestring)}
            if renamed:
                self.data = self.data.rename(columns=renamed)
            self.variables = indexes + list(self.data.columns)

            # Reset groupby dimensions
            groupby_index = [g for g in groupby if g in indexes]
            if groupby_index:
                self.data = self.data.reset_index(groupby_index)
            not_found = [g for g in groupby if g not in list(self.data.columns) + indexes]
            if groupby and not_found:
                raise ValueError('The supplied groupby dimension(s) %s '
                                 'could not be found, expected one or '
                                 'more of: %s' % (not_found, list(self.data.columns)))
        # Set data-level options
        self.x = x
        self.y = y
        self.kind = kind or 'line'
        self.datatype = datatype
        self.gridded = gridded
        self.use_dask = use_dask
        self.indexes = indexes
        if isinstance(by, (np.ndarray, pd.Series)):
            self.data['by'] = by
            self.by = ['by']
        elif not by:
            self.by = []
        else:
            self.by = by if isinstance(by, list) else [by]
        self.groupby = groupby
        self.hover_cols = hover_cols

    def _process_style(self, colormap, kwds):
        plot_options = {}
        kind = self.kind
        eltype = self._kind_mapping[kind]
        if eltype in Store.registry['bokeh']:
            valid_opts = Store.registry['bokeh'][eltype].style_opts
        else:
            valid_opts = []

        for opt in valid_opts:
            if opt not in kwds or not isinstance(kwds[opt], list) or opt == 'cmap':
                continue
            kwds[opt] = Cycle(kwds[opt])

        style_opts = {kw: kwds[kw] for kw in list(kwds) if kw in valid_opts}

        # Process style options
        if 'cmap' in kwds and colormap:
            raise TypeError("Only specify one of `cmap` and `colormap`.")
        elif 'cmap' in kwds:
            cmap = kwds.pop('cmap')
        else:
            cmap = colormap

        if kind.startswith('bar'):
            plot_options['stacked'] = self.stacked
        if 'color' in kwds or 'c' in kwds:
            color = kwds.pop('color', kwds.pop('c', None))
            if isinstance(color, (np.ndarray, pd.Series)):
                self.data['_color'] = color
                kwds['c'] = '_color'
            elif isinstance(color, list):
                style_opts['color'] = color
            else:
                style_opts['color'] = color
                if 'c' in self._kind_options.get(kind, []) and (color in self.variables):
                    kwds['c'] = color
                    if self.data[color].dtype.kind in 'OSU':
                        cmap = cmap or 'Category10'
                    else:
                        plot_options['colorbar'] = True
        if 'size' in kwds or 's' in kwds:
            size = kwds.pop('size', kwds.pop('s', None))
            if isinstance(size, (np.ndarray, pd.Series)):
                self.data['_size'] = np.sqrt(size)
                kwds['s'] = '_size'
            elif isinstance(size, hv.util.basestring):
                kwds['s'] = size
                if 'scale' in kwds:
                    style_opts['size'] = kwds['scale']
            elif not isinstance(size, dim):
                style_opts['size'] = np.sqrt(size)
        elif 'size' in valid_opts:
            style_opts['size'] = np.sqrt(30)

        if cmap:
            style_opts['cmap'] = cmap

        return style_opts, plot_options, kwds

    def _validate_kwds(self, kwds):
        kind_opts = self._kind_options.get(self.kind, [])
        kind = self.kind
        eltype = self._kind_mapping[kind]
        if eltype in Store.registry['bokeh']:
            valid_opts = Store.registry['bokeh'][eltype].style_opts
        ds_opts = ['max_px', 'threshold']
        mismatches = sorted([k for k in kwds if k not in kind_opts + ds_opts + valid_opts])
        if not mismatches:
            return

        if 'ax' in mismatches:
            mismatches.pop(mismatches.index('ax'))
            warnings.warn('hvPlot does not have the concept of axes, '
                          'and the ax keyword will be ignored. Compose '
                          'plots with the * operator to overlay plots or the '
                          '+ operator to lay out plots beside each other '
                          'instead.')
        if 'figsize' in mismatches:
            mismatches.pop(mismatches.index('figsize'))
            warnings.warn('hvPlot does not have the concept of a figure, '
                          'and the figsize keyword will be ignored. The '
                          'size of each subplot in a layout is set '
                          'individually using the width and height options.')

        combined_opts = (self._data_options + self._axis_options +
                         self._style_options + self._op_options + kind_opts +
                         valid_opts)
        for mismatch in mismatches:
            suggestions = difflib.get_close_matches(mismatch, combined_opts)
            warnings.warn('%s option not found for %s plot; similar options '
                          'include: %r' % (mismatch, self.kind, suggestions))

    def __call__(self, kind, x, y):
        kind = self.kind or kind
        method = getattr(self, kind)

        groups = self.groupby
        zs = self.kwds.get('z', [])
        if not isinstance(zs, list): zs = [zs]
        grid = []
        if self.row: grid.append(self.row)
        if self.col: grid.append(self.col)
        groups += grid
        if groups or len(zs) > 1:
            if self.streaming:
                raise NotImplementedError("Streaming and groupby not yet implemented")
            data = self.data
            if not self.gridded and any(g in self.indexes for g in groups):
                data = data.reset_index()

            dataset = Dataset(data)
            if groups:
                dataset = dataset.groupby(groups, dynamic=self.dynamic)
                if len(zs) > 1:
                    dimensions = [Dimension(self.group_label, values=zs)] + dataset.kdims
                    if self.dynamic:
                        obj = DynamicMap(lambda *args: getattr(self, kind)(x, y, args[0], dataset[args[1:]].data),
                                         kdims=dimensions)
                    else:
                        obj = HoloMap({(z,) + k: getattr(self, kind)(x, y, z, dataset[k])
                                       for k, v in dataset.data.items() for z in zs}, kdims=dimensions)
                else:
                    obj = dataset.map(lambda ds: getattr(self, kind)(x, y, data=ds.data), Dataset)
            elif len(zs) > 1:
                if self.dynamic:
                    dataset = DynamicMap(lambda z: getattr(self, kind)(x, y, z, data=dataset.data),
                                         kdims=[Dimension(self.group_label, values=zs)])
                else:
                    dataset = HoloMap({z: getattr(self, kind)(x, y, z, data=dataset.data) for z in zs},
                                      kdims=[self.group_label])
            else:
                obj = getattr(self, kind)(x, y, data=dataset.data)
            if grid:
                obj = obj.grid(grid).options(shared_xaxis=True, shared_yaxis=True)
        else:
            obj = method(x, y)

        if not (self.datashade or self.rasterize):
            return obj

        try:
            from holoviews.operation.datashader import datashade, rasterize, dynspread
            from datashader import count_cat
        except:
            raise ImportError('Datashading is not available')

        opts = dict(width=self._plot_opts['width'], height=self._plot_opts['height'],
                    dynamic=self.dynamic)
        if 'cmap' in self._style_opts and self.datashade:
            levels = self._plot_opts.get('color_levels')
            cmap = self._style_opts['cmap']
            if isinstance(cmap, dict):
                opts['color_key'] = cmap
            else:
                opts['cmap'] = process_cmap(cmap, levels)

        if self.by:
            opts['aggregator'] = count_cat(self.by[0])
        if self.aggregator:
            opts['aggregator'] = self.aggregator
        if self.precompute:
            opts['precompute'] = self.precompute
        if self.x_sampling:
            opts['x_sampling'] = self.x_sampling
        if self.y_sampling:
            opts['y_sampling'] = self.y_sampling

        style = {}
        if self.datashade:
            operation = datashade
            eltype = 'RGB'
        else:
            operation = rasterize
            eltype = 'Image'
            if 'cmap' in self._style_opts:
                style['cmap'] = self._style_opts['cmap']

        processed = operation(obj, **opts)

        if self.dynspread:
            if self.datashade:
                processed = dynspread(processed, max_px=self.kwds.get('max_px', 3),
                                      threshold=self.kwds.get('threshold', 0.5))
            else:
                warnings.warn('dynspread may only be applied on datashaded plots, '
                              'use datashade=True instead of rasterize=True.')
        return processed.opts({eltype: {'plot': self._plot_opts, 'style': style}})

    def dataset(self, x=None, y=None, data=None):
        data = self.data if data is None else data
        return Dataset(data, self.kwds.get('columns'), []).redim(**self._redim)

    ##########################
    #     Simple charts      #
    ##########################

    def single_chart(self, element, x, y, data=None):
        labelled = ['y' if self.invert else 'x'] if x != 'index' else []
        if not self.is_series:
            labelled.append('x' if self.invert else 'y')
        elif not self.label:
            self._relabel['label'] = y

        if 'xlabel' in self._plot_opts and 'x' not in labelled:
            labelled.append('x')
        if 'ylabel' in self._plot_opts and 'y' not in labelled:
            labelled.append('y')

        opts = {element.__name__: dict(
            plot=dict(self._plot_opts, labelled=labelled),
            norm=self._norm_opts, style=self._style_opts
        )}

        data = self.data if data is None else data
        ys = [y]
        if element is Area and self.kwds.get('y2'):
            ys += [self.kwds['y2']]
        for p in 'cs':
            if p in self.kwds and self.kwds[p] in data.columns:
                ys += [self.kwds[p]]
        ys += self.hover_cols

        if self.by:
            if element is Bars and not self.subplots:
                return element(data, [x] + self.by, ys).relabel(**self._relabel).redim(**self._redim).opts(opts)
            chart = Dataset(data, self.by + [x], ys).to(element, x, ys, self.by).relabel(**self._relabel)
            chart = chart.layout() if self.subplots else chart.overlay().options(batched=False)
        else:
            chart = element(data, x, ys).relabel(**self._relabel)
        return chart.redim(**self._redim).opts(opts)

    def _process_args(self, data, x, y):
        data = (self.data if data is None else data)
        x = x or self.x
        if not x and self.use_index:
            x = self.indexes[0]
        elif not x:
            raise ValueError('Could not determine what to plot. Expected '
                             'x to be declared or use_index to be enabled.')

        y = y or self.y
        if not y:
            ys = [c for c in data.columns if c not in [x] + self.by + self.groupby]
            y = ys[0] if len(ys) == 1 else ys
        return data, x, y

    def chart(self, element, x, y, data=None):
        "Helper method for simple x vs. y charts"
        data, x, y = self._process_args(data, x, y)
        if x and y and not isinstance(y, (list, tuple)):
            return self.single_chart(element, x, y, data)
        elif x and y and len(y) == 1:
            return self.single_chart(element, x, y[0], data)

        labelled = ['y' if self.invert else 'x'] if x != 'index' else []
        if self.value_label != 'value':
            labelled.append('x' if self.invert else 'y')

        if 'xlabel' in self._plot_opts and 'x' not in labelled:
            labelled.append('x')
        if 'ylabel' in self._plot_opts and 'y' not in labelled:
            labelled.append('y')

        opts = dict(plot=dict(self._plot_opts, labelled=labelled),
                    norm=self._norm_opts, style=self._style_opts)
        charts = []
        for c in y:
            chart = element(data, x, [c] + self.hover_cols).redim(**{c: self.value_label})
            charts.append((c, chart.relabel(**self._relabel).opts(**opts)))
        return self._by_type(charts, self.group_label, sort=False).options('NdOverlay', batched=False)

    def line(self, x, y, data=None):
        return self.chart(Curve, x, y, data)

    def step(self, x, y, data=None):
        where = self.kwds.get('where', 'mid')
        return self.line(x, y, data).options('Curve', interpolation='steps-' + where)

    def scatter(self, x, y, data=None):
        scatter = self.chart(Scatter, x, y, data)
        opts = {}
        if 's' in self.kwds:
            opts['size_index'] = self.kwds['s']
        if 'marker' in self.kwds:
            opts['marker'] = self.kwds['marker']
        return scatter.options('Scatter', **opts) if opts else scatter

    def area(self, x, y, data=None):
        areas = self.chart(Area, x, y, data)
        if self.stacked:
            areas = areas.map(Area.stack, NdOverlay)
        return areas

    ##########################
    #  Categorical charts    #
    ##########################

    def _category_plot(self, element, x, y, data):
        """
        Helper method to generate element from indexed dataframe.
        """
        labelled = ['y' if self.invert else 'x'] if x != 'index' else []
        if self.value_label != 'value':
            labelled.append('x' if self.invert else 'y')

        if 'xlabel' in self._plot_opts and 'x' not in labelled:
            labelled.append('x')
        if 'ylabel' in self._plot_opts and 'y' not in labelled:
            labelled.append('y')

        opts = {'plot': dict(self._plot_opts, labelled=labelled),
                'style': dict(self._style_opts),
                'norm': self._norm_opts}

        id_vars = [x]
        if any(v in self.indexes for v in id_vars):
            data = data.reset_index()
        data = data[y + [x]]

        if check_library(data, 'dask'):
            from dask.dataframe import melt
        else:
            melt = pd.melt

        df = melt(data, id_vars=[x], var_name=self.group_label, value_name=self.value_label)
        kdims = [x, self.group_label]
        vdims = [self.value_label] + self.hover_cols
        if self.subplots:
            obj = Dataset(df, kdims, vdims).to(element, x).layout()
        else:
            obj = element(df, kdims, vdims)
        return obj.redim(**self._redim).relabel(**self._relabel).opts(**opts)

    def bar(self, x, y, data=None):
        data, x, y = self._process_args(data, x, y)
        if x and y and (self.by or not isinstance(y, (list, tuple) or len(y) == 1)):
            y = y[0] if isinstance(y, (list, tuple)) else y
            return self.single_chart(Bars, x, y, data)
        return self._category_plot(Bars, x, list(y), data)

    def barh(self, x, y, data=None):
        return self.bar(x, y, data).opts(plot={'Bars': dict(invert_axes=True)})

    ##########################
    #   Statistical charts   #
    ##########################

    def _stats_plot(self, element, y, data=None):
        """
        Helper method to generate element from indexed dataframe.
        """
        data, x, y = self._process_args(data, None, y)

        opts = {'plot': dict(self._plot_opts), 'norm': self._norm_opts,
                'style': self._style_opts}

        ylim = self._plot_opts.get('ylim', (None, None))
        if not isinstance(y, (list, tuple)):
            ranges = {y: ylim}
            return (element(data, self.by, y).redim.range(**ranges).relabel(**self._relabel).opts(**opts))

        labelled = ['y' if self.invert else 'x'] if self.group_label != 'Group' else []
        if self.value_label != 'value':
            labelled.append('x' if self.invert else 'y')

        if 'xlabel' in self._plot_opts and 'x' not in labelled:
            labelled.append('x')
        if 'ylabel' in self._plot_opts and 'y' not in labelled:
            labelled.append('y')

        opts['plot']['labelled'] = labelled

        kdims = [self.group_label]
        data = data[list(y)]
        if check_library(data, 'dask'):
            from dask.dataframe import melt
        else:
            melt = pd.melt
        df = melt(data, var_name=self.group_label, value_name=self.value_label)
        ranges = {self.value_label: ylim}
        return (element(df, kdims, self.value_label).redim(**self._redim)
                .redim.range(**ranges).relabel(**self._relabel).opts(**opts))

    def box(self, x, y, data=None):
        return self._stats_plot(BoxWhisker, y, data)

    def violin(self, x, y, data=None):
        try:
            from holoviews.element import Violin
        except ImportError:
            raise ImportError('Violin plot requires HoloViews version >=1.10')
        return self._stats_plot(Violin, y, data)

    def hist(self, x, y, data=None):
        data, x, y = self._process_args(data, x, y)

        labelled = ['y'] if self.invert else ['x']

        if 'xlabel' in self._plot_opts and 'x' not in labelled:
            labelled.append('x')
        if 'ylabel' in self._plot_opts and 'y' not in labelled:
            labelled.append('y')

        plot_opts = dict(self._plot_opts, labelled=labelled)
        opts = dict(plot=plot_opts, style=self._style_opts,
                    norm=self._norm_opts)

        hist_opts = {'bin_range': self.kwds.get('bin_range', None),
                     'normed': self.kwds.get('normed', False),
                     'cumulative': self.kwds.get('cumulative', False)}
        if 'bins' in self.kwds:
            bins = self.kwds['bins']
            if isinstance(bins, int):
                hist_opts['num_bins'] = bins
            else:
                hist_opts['bins'] = bins

        if not isinstance(y, (list, tuple)):
            if not 'bin_range' in self.kwds:
                ys = data[y]
                ymin, ymax = (ys.min(), ys.max())
                if is_dask(ys):
                    ymin, ymax = ymin.compute(), ymax.compute()
                hist_opts['bin_range'] = ymin, ymax

            ds = Dataset(data, self.by, y)
            hist = hists = histogram(ds.to(Dataset, [], y, self.by), **hist_opts)
            if self.by:
                hist = hists.last
                hists = hists.layout() if self.subplots else hists.overlay()
            return hists.opts({'Histogram': opts}).redim(**self._redim)

        ranges = []
        for col in y:
            if not 'bin_range' in self.kwds:
                ys = data[col]
                ymin, ymax = (ys.min(), ys.max())
                if is_dask(ys):
                    ymin, ymax = ymin.compute(), ymax.compute()
                ranges.append((ymin, ymax))
        if ranges:
            hist_opts['bin_range'] = max_range(ranges)

        ds = Dataset(data)
        hists = []
        for col in y:
            hist = histogram(ds, dimension=col, **hist_opts)
            hists.append((col, hist.relabel(**self._relabel).opts(**opts)))
        return self._by_type(hists, sort=False).redim(**self._redim)

    def kde(self, x, y, data=None):
        data, x, y = self._process_args(data, x, y)
        opts = dict(plot=self._plot_opts, style=self._style_opts, norm=self._norm_opts)
        opts = {'Distribution': opts, 'Area': opts,
                'NdOverlay': {'plot': dict(self._plot_opts, legend_limit=0)}}

        xlim = self._plot_opts.get('xlim', (None, None))
        if not isinstance(y, (list, tuple)):
            ranges = {y: xlim}
            if self.by:
                dists = Dataset(data).to(Distribution, y, [], self.by)
                dists = dists.layout() if self.subplots else dists.overlay()
            else:
                dists = Distribution(data, y, [])
        else:
            ranges = {self.value_label: xlim}
            data = data[y]
            df = pd.melt(data, var_name=self.group_label, value_name=self.value_label)
            ds = Dataset(df)
            if len(df):
                dists = ds.to(Distribution, self.value_label).overlay()
            else:
                vdim = self.value_label + ' Density'
                dists = NdOverlay({0: Area([], self.value_label, vdim)},
                                  [self.group_label])
        return dists.redim(**self._redim).redim.range(**ranges).relabel(**self._relabel).opts(opts)

    ##########################
    #      Other charts      #
    ##########################

    def heatmap(self, x, y, data=None):
        data = self.data if data is None else data
        if not x: x = self.x or data.columns[0]
        if not y: y = self.y or data.columns[1]
        z = self.kwds.get('C', [c for c in data.columns if c not in (x, y)][0])
        z = [z] + self.hover_cols

        opts = dict(plot=self._plot_opts, norm=self._norm_opts, style=self._style_opts)
        hmap = HeatMap(data, [x, y], z).redim(**self._redim).opts(**opts)
        if 'reduce_function' in self.kwds:
            return hmap.aggregate(function=self.kwds['reduce_function'])
        return hmap

    def hexbin(self, x, y, data=None):
        data = self.data if data is None else data
        if not x: x = data.columns[0]
        if not y: y = data.columns[1]
        z = [self.kwds['C']] if self.kwds.get('C') else []
        z += self.hover_cols

        opts = dict(plot=self._plot_opts, norm=self._norm_opts, style=self._style_opts)
        if 'reduce_function' in self.kwds:
            opts['plot']['aggregator'] = self.kwds['reduce_function']
        if 'gridsize' in self.kwds:
            opts['plot']['gridsize'] = self.kwds['gridsize']
        return HexTiles(data, [x, y], z or []).redim(**self._redim).opts(**opts)

    def bivariate(self, x, y, data=None):
        data = self.data if data is None else data
        if not x: x = data.columns[0]
        if not y: y = data.columns[1]

        opts = dict(plot=self._plot_opts, norm=self._norm_opts, style=self._style_opts)
        return Bivariate(data, [x, y]).redim(**self._redim).opts(**opts)

    def table(self, x=None, y=None, data=None):
        allowed = ['width', 'height']
        opts = {k: v for k, v in self._plot_opts.items() if k in allowed}
        data = self.data if data is None else data
        return Table(data, self.kwds.get('columns'), []).redim(**self._redim).opts(plot=opts)

    def labels(self, x, y, data=None):
        data = self.data if data is None else data
        if not x: x = self.x or data.columns[0]
        if not y: y = self.y or data.columns[1]
        text = self.kwds.get('text', [c for c in data.columns if c not in (x, y)][0])
        vdims = [text]
        style = self._style_opts
        if 'c' in self.kwds:
            self._plot_opts['color_index'] = self.kwds['c']
            vdims.append(self.kwds['c'])
            style.pop('color', None)
        if 's' in self.kwds:
            self._plot_opts['size_index'] = self.kwds['s']
            vdims.append(self.kwds['s'])
        opts = dict(plot=self._plot_opts, norm=self._norm_opts, style=style)
        return Labels(data, [x, y], vdims).redim(**self._redim).opts(**opts)

    ##########################
    #     Gridded plots      #
    ##########################

    def _get_element(self, kind):
        element = self._kind_mapping[kind]
        return element

    def image(self, x=None, y=None, z=None, data=None):

        data = self.data if data is None else data

        z = z or self.kwds.get('z')
        x = x or self.x
        y = y or self.y
        if not (x and y):
            x, y = data.coords[::-1]
        if not z:
            z = data.title if isinstance(data, NDDataset) else list(data.data_vars)[0]
        z = [z] + self.hover_cols

        params = dict(self._relabel)
        opts = dict(plot=self._plot_opts, style=self._style_opts, norm=self._norm_opts)
        ranges = {z[0]: self._dim_ranges['c']}

        element = self._get_element('image')

        params['datatype'] = [self.datatype]
        return element(data, [x, y], z, **params).redim(**self._redim).redim.range(**ranges).opts(**opts)

    def rgb(self, x=None, y=None, data=None):
        data = self.data if data is None else data

        coords = [c for c in data.coords if c not in self.groupby + self.by]
        if len(coords) < 3:
            raise ValueError('Data must be 3D array to be converted to RGB.')
        x = x or coords[2]
        y = y or coords[1]
        bands = self.kwds.get('bands', coords[0])
        z = self.kwds.get('z')
        if z is None:
            z = list(data.data_vars)[0]
        data = data[z]
        nbands = len(data.coords[bands])
        if nbands < 3:
            raise ValueError('Selected bands coordinate (%s) has only %d channels,'
                             'expected at least three channels to convert to RGB.' %
                             (bands, nbands))

        params = dict(self._relabel)
        opts = dict(plot=self._plot_opts, style=self._style_opts, norm=self._norm_opts)
        xres, yres = data.attrs['res'] if 'res' in data.attrs else (1, 1)
        xs = data.coords[x][::-1] if xres < 0 else data.coords[x]
        ys = data.coords[y][::-1] if yres < 0 else data.coords[y]
        eldata = (xs, ys)
        for b in range(nbands):
            eldata += (data.isel(**{bands: b}).values,)
        rgb = RGB(eldata, [x, y], RGB.vdims[:nbands], **params)
        return rgb.redim(**self._redim).opts(**opts)

    def quadmesh(self, x=None, y=None, z=None, data=None):
        import xarray as xr
        data = self.data if data is None else data

        z = z or self.kwds.get('z')
        x = x or self.x
        y = y or self.y
        if not (x and y):
            x, y = list([k for k, v in data.coords.items() if v.size > 1])
        if not z:
            z = list(data.data_vars)[0] if isinstance(data, xr.Dataset) else data.name
        z = [z] + self.hover_cols

        params = dict(self._relabel)
        opts = dict(plot=self._plot_opts, style=self._style_opts, norm=self._norm_opts)
        ranges = {z[0]: self._dim_ranges['c']}

        element = self._get_element('quadmesh')
        return element(data, [x, y], z, **params).redim(**self._redim).redim.range(**ranges).opts(**opts)

    def contour(self, x=None, y=None, z=None, data=None, filled=False):
        from holoviews.operation import contours

        opts = dict(plot=self._plot_opts, style=self._style_opts, norm=self._norm_opts)
        qmesh = self.quadmesh(x, y, z, data)

        if filled:
            opts['style']['line_alpha'] = 0

        if opts['plot']['colorbar']:
            opts['plot']['show_legend'] = False
        levels = self.kwds.get('levels', 5)
        if isinstance(levels, int):
            opts['plot']['color_levels'] = levels
        return contours(qmesh, filled=filled, levels=levels).opts(**opts)

    def contourf(self, x=None, y=None, z=None, data=None):
        return self.contour(x, y, z, data, filled=True)

    def points(self, x=None, y=None, data=None):
        data = self.data if data is None else data
        params = dict(self._relabel)

        x = x or self.x
        y = y or self.y

        plot_opts = dict(self._plot_opts)
        ranges = {}
        if 'c' in self.kwds:
            plot_opts['color_index'] = self.kwds['c']
            self._style_opts.pop('color', None)
            ranges[self.kwds['c']] = self._dim_ranges['c']
        if 's' in self.kwds:
            plot_opts['size_index'] = self.kwds['s']
        if 'marker' in self.kwds:
            plot_opts['marker'] = self.kwds['marker']
        opts = dict(plot=plot_opts, style=self._style_opts, norm=self._norm_opts)
        element = self._get_element('points')
        vdims = [self.kwds['c']] if 'c' in self.kwds else []
        if 's' in self.kwds:
            vdims.append(self.kwds['s'])
        vdims = vdims + self.hover_cols
        params['vdims'] = vdims
        return element(data, [x, y], **params).redim(**self._redim).redim.range(**ranges).opts(**opts)
