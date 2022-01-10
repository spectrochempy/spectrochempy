# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

"""Define plots."""

# import plotly.graph_objects as go
# import numpy as np
#
# from spectrochempy.utils import colorscale
# from spectrochempy.core import preferences
#
# # from matplotlib.ticker import MaxNLocator
#
# __all__ = ['plotly', 'plotly_stack']
# __dataset_methods__ = __all__
#
#
# # ======================================================================================================================
# # nddataset plot2D functions
# # ======================================================================================================================
#
# # contour map (default) -------------------------------------------------------
#
# def plotly_map(dataset, **kwargs):
#     """
#     Plot a 2D dataset as a contoured map.
#
#     Alias of plot_2D (with `method` argument set to ``map``.
#     """
#     kwargs['method'] = 'map'
#     return plotly(dataset, **kwargs)
#
#
# # stack plot  -----------------------------------------------------------------
#
# def plotly_stack(dataset, **kwargs):
#     """
#     Plot a 2D dataset as a stacked plot.
#
#     Alias of plot_2D (with `method` argument set to ``stack``).
#     """
#     kwargs['method'] = 'stack'
#     return plotly(dataset, **kwargs)
#
#
# # image plot --------------------------------------------------------
#
# def plotly_image(dataset, **kwargs):
#     """
#     Plot a 2D dataset as an image plot.
#
#     Alias of plot_2D (with `method` argument set to ``image``).
#     """
#     kwargs['method'] = 'image'
#     return plotly(dataset, **kwargs)
#
#
# # surface plot -----------------------------------------------------------------
#
# def plotly_surface(dataset, **kwargs):
#     """
#     Plot a 2D dataset as a a 3D-surface.
#
#     Alias of plot_2D (with `method` argument set to ``surface``.
#     """
#     kwargs['method'] = 'surface'
#     return plotly(dataset, **kwargs)
#
#
# # generic plot (default stack plot) -------------------------------------------
#
# def plotly(dataset, **kwargs):
#     """
#     Generate a Plotly plot
#
#     Parameters
#     ----------
#     dataset; |NDDataset|
#         The dataset to plot
#     kwargs: any
#         Additional keyword arguments
#
#     Returns
#     -------
#     figure
#         A plotly figure
#     """
#
#     # TODO: not finished    (replace preferences)
#     # get all plot preferences
#     # ------------------------------------------------------------------------
#
#     prefs = dataset.preferences
#
#     # method of plot
#     # ------------------------------------------------------------------------
#     method = kwargs.get('method', None)
#
#     if not prefs.style:
#         # not yet set, initialize with default project preferences
#         prefs.update(dataset_preferences.to_dict())
#
#         # surface specific setting
#         if method not in ['surface']:
#             prefs['colorbar'] = False
#
#     if method is None:
#         method = prefs.method_2D
#
#     selector = kwargs.get('selector', '[Processed]')
#     data_transposed = True if 'Transposed' in selector else False
#
#     name = dataset.name
#     if data_transposed:
#         new = dataset.copy().T  # transpose dataset
#         nameadd = ' (Transposed)'
#     else:
#         new = dataset
#         nameadd = ''
#
#     # new = new.squeeze()
#
#     # ------------------------------------------------------------------------
#     # coordinates
#     # ------------------------------------------------------------------------
#
#     # the actual dimension name for x is the last in the new.dims list
#     dimx = new.dims[-1]
#     x = getattr(new, dimx)
#
#     # the actual dimension name for y is the one before last in the new.dims list
#     dimy = new.dims[-2]
#     y = getattr(new, dimy)
#
#     # ------------------------------------------------------------------------
#     # Should we display only ROI region?
#     # ------------------------------------------------------------------------
#
#     if 'Processed' in selector:
#         # in this case we make the selection
#         new = new[y.roi[0]:y.roi[1], x.roi[0]:x.roi[1]]
#         x = getattr(new, dimx)
#         y = getattr(new, dimy)
#
#     xsize = new.shape[-1]
#     ysize = new.shape[-2]
#
#     # figure setup
#     # ------------------------------------------------------------------------
#     fig = new._fig
#
#     # should we use the previous figure?
#     clear = kwargs.get('clear', True)
#
#     dragmode = kwargs.get('dragmode', 'zoom')
#
#     if fig is None or not isinstance(fig, go.Figure) or clear:
#
#         fig = go.Figure()
#
#         # set the layout
#         layout = dict(
#                 title=name + nameadd,
#                 paper_bgcolor='rgba(0,0,0,0)',  # transparent
#                 autosize=True,
#                 hovermode='closest',
#                 showlegend=False,
#                 clickmode='event+select',
#                 dragmode=dragmode,
#                 selectdirection='h',
#                 margin=dict(t=43, r=50),
#                 )
#
#         fig.update_layout(layout)
#
#     if dragmode == 'select':
#         fig.update_layout(
#                 paper_bgcolor='lightsalmon',
#                 annotations=[
#                         dict(
#                                 x=2,
#                                 y=5,
#                                 xref="x",
#                                 yref="y",
#                                 text="Mask selection mode ACTIVE",
#
#                                 ax=0,
#                                 ay=-40
#                                 )
#                         ]
#                 )
#     # Other properties
#     # ------------------------------------------------------------------------
#
#     # colorbar = kwargs.get('colorbar', prefs.colorbar)
#
#     cmap = kwargs.get('cmap', 'viridis')
#
#     # viridis is the default setting,
#     # so we assume that it must be overwritten here
#     # except if style is grayscale which is a particular case.
#     style = kwargs.get('style', prefs.style)
#
#     if style and "grayscale" not in style and cmap == "viridis":
#
#         if method in ['map', 'image']:
#             cmap = kwargs.get('colormap',
#                               kwargs.get('cmap', prefs.colormap))
#         elif data_transposed:
#             cmap = kwargs.get('colormap',
#                               kwargs.get('cmap', prefs.colormap_transposed))
#         elif method in ['surface']:
#             cmap = kwargs.get('colormap',
#                               kwargs.get('cmap', prefs.colormap_surface))
#         else:
#             cmap = kwargs.get('colormap',
#                               kwargs.get('cmap', prefs.colormap_stack))
#
#     # lw = kwargs.get('linewidth', kwargs.get('lw',
#     #                                       prefs.pen_linewidth))
#
#     # alpha = kwargs.get('calpha', prefs.contour_alpha)
#
#     # antialiased = kwargs.get('antialiased', prefs.antialiased)
#
#     # rcount = kwargs.get('rcount', prefs.rcount)
#
#     # ccount = kwargs.get('ccount', prefs.ccount)
#
#     number_x_labels = prefs.number_of_x_labels
#     # number_y_labels = prefs.number_of_y_labels
#
#     # ax.xaxis.set_major_locator(MaxNLocator(nbins=number_x_labels))
#     # ax.yaxis.set_major_locator(MaxNLocator(nbins=number_y_labels))
#     # x_locator = MaxNLocator(nbins=number_x_labels)
#     # y_locator = MaxNLocator(nbins=number_y_labels)
#
#     # if method not in ['surface']:
#     #    ax.xaxis.set_ticks_position('bottom')
#     #    ax.yaxis.set_ticks_position('left')
#
#     # the next lines are to avoid multipliers in axis scale
#     # formatter = ScalarFormatter(useOffset=False)
#     # ax.xaxis.set_major_formatter(formatter)
#     # ax.yaxis.set_major_formatter(formatter)
#
#     # ------------------------------------------------------------------------
#     # Set axis
#     # ------------------------------------------------------------------------
#
#     # set the abscissa axis
#     # ------------------------------------------------------------------------
#
#     # discrete_data = False
#     if x is not None and (not x.is_empty or x.is_labeled):
#         xdata = x.data
#         if not np.any(xdata):
#             if x.is_labeled:
#                 # discrete_data = True
#                 # take into account the fact that sometimes axis have just labels
#                 xdata = range(1, len(x.labels) + 1)
#             # TODO it would be more convenient if the data attribute returned the correct values
#     else:
#         xdata = range(xsize)
#
#     xl = [xdata[0], xdata[-1]]
#     xl.sort()
#
#     if xsize < number_x_labels + 1:
#         # extend the axis so that the labels are not too close to the limits
#         inc = abs(xdata[1] - xdata[0]) * .5
#         xl = [xl[0] - inc, xl[1] + inc]
#
#     xlim = list(kwargs.get('xlim', xl))  # zoom?
#     xlim.sort()
#     xlim[-1] = min(xlim[-1], xl[-1])
#     xlim[0] = max(xlim[0], xl[0])
#
#     if kwargs.get('x_reverse', kwargs.get('reverse', x.reversed if x else False)):
#         xlim.reverse()
#
#     # xscale = kwargs.get("xscale", "linear")
#
#     fig.update_layout(
#             dict(
#                     xaxis=_make_axis('x', range=xlim, label=f'{x.alt_title} / {x.units:~P}', **kwargs),
#                     )
#             )
#
#     # set the ordinates axis
#     # ------------------------------------------------------------------------
#     # the actual dimension name is the second in the new.dims list
#     dimy = new.dims[-2]
#     y = getattr(new, dimy)
#     ysize = new.shape[-2]
#     if y is not None and (not y.is_empty or y.is_labeled):
#         ydata = y.data
#         if not np.any(ydata):
#             if y.is_labeled:
#                 ydata = range(1, len(y.labels) + 1)
#     else:
#         ydata = range(ysize)
#
#     yl = [ydata[0], ydata[-1]]
#     yl.sort()
#
#     # if ysize < number_y_labels + 1:
#     #    # extend the axis so that the labels are not too close to the limits
#     #    inc = abs(ydata[1] - ydata[0]) * .5
#     #    yl = [yl[0] - inc, yl[1] + inc]
#
#     ylim = list(kwargs.get("ylim", yl))
#     ylim.sort()
#     ylim[-1] = min(ylim[-1], yl[-1])
#     ylim[0] = max(ylim[0], yl[0])
#
#     # yscale = kwargs.get("yscale", "linear")
#
#     fig.update_layout(
#             dict(
#                     yaxis=_make_axis('y', label=f'{new.title} / {new.units:~P}', **kwargs),
#                     )
#             )
#
#     zoomreset = kwargs.get('zoomreset', None)
#     uirevision = f'{new}  {zoomreset}'
#     fig.update_layout(
#             dict(
#                     uirevision=uirevision,
#                     )
#             )
#
#     # Data
#
#     # colorscale
#     amp = 0
#     mi = np.ma.min(y.data)
#     ma = np.ma.max(y.data)
#
#     if kwargs.get('reduce_range', True):
#         amp = (ma - mi) / 20.
#
#     if kwargs.get('extrema', True):
#         vmin, vmax = mi - amp, ma + amp
#     else:
#         vmin, vmax = -ma - amp, ma + amp
#
#     # Here I am using matplotlib to find the color map
#     # because I did not find the way to do this easily using plotly)
#     colorscale.normalize(vmin, vmax, cmap=cmap, rev=False)
#
#     # DATA
#
#     data = []
#
#     optimisation = kwargs.get('optimisation', 0)
#     offset = getattr(new, dimx).offset
#
#     for trace in _compress(new[::-1], optimisation):
#         name = f'{getattr(trace, dimy).values:~P}'  # - dataset.y.offset_value:~P}'
#         color = colorscale.rgba(getattr(trace, dimy).data)  # , offset=yoffset)
#         trace = trace.squeeze()
#         x = getattr(trace, dimx).data
#         if trace.mask is not None and np.any(trace.mask):
#             z = trace.data
#             z[trace.mask] = np.NaN
#         else:
#             z = trace.data
#         y_string = f'{getattr(new, dimy).alt_title}:' + ' %{fullData.name} <br>' if getattr(new, dimy).size > 1 else ''
#         data.append(dict(x=x,
#                          xaxis='x',
#                          y=z,
#                          yaxis='y',
#                          name=name,
#                          hoverlabel=dict(
#                                  font_size=12,
#                                  font_family="Rockwell"
#                                  ),
#                          hovertemplate=f'{getattr(trace, dimx).alt_title}:'
#                                        ' %{x:.2f} '
#                                        f'{getattr(trace, dimx).units:~P}'
#                                        f'<br>'
#                                        f'{y_string}'
#                                        f'{trace.alt_title}:'
#                                        ' %{y:.2f} '
#                                        f'{trace.units:~P}'
#                                        '<extra></extra>',
#                          mode='lines',
#                          type='scattergl',
#                          connectgaps=False,
#                          line=dict(
#                                  color=color,
#                                  dash='solid',
#                                  width=1.5),
#                          )
#                     )
#     fig.add_traces(data)
#
#     # show out of X ROI zone
#     fullrange = getattr(new, dimx).limits
#     roirange = getattr(new, dimx).roi
#
#     x0, x1 = fullrange[0], roirange[0] - offset
#     x2, x3 = fullrange[1], roirange[1] - offset
#     fig.update_layout(
#             shapes=[
#                     dict(
#                             type="rect",
#                             # x-reference is assigned to the x-values
#                             xref="x",
#                             # y-reference is assigned to the plot paper [0,1]
#                             yref="paper",
#                             x0=x0,
#                             y0=0,
#                             x1=x1,
#                             y1=1,
#                             fillcolor="LightSalmon",
#                             opacity=0.2,
#                             layer="below",
#                             line_width=0,
#                             ),
#                     dict(
#                             type="rect",
#                             # x-reference is assigned to the x-values
#                             xref="x",
#                             # y-reference is assigned to the plot paper [0,1]
#                             yref="paper",
#                             x0=x2,
#                             y0=0,
#                             x1=x3,
#                             y1=1,
#                             fillcolor="LightSalmon",
#                             opacity=0.2,
#                             layer="below",
#                             line_width=0,
#                             ),
#                     ]
#             )
#
#     return fig
#
#
# def _make_axis(axis,
#                range=None,
#                label=None,
#                **kwargs):
#     fontsize = kwargs.get('fontsize', 18)
#
#     return dict(
#             anchor='y' if axis == 'x' else 'x',
#             domain=[0.0, 1.0],
#             nticks=7,
#             range=range,
#             showgrid=True,
#             side='bottom' if axis == 'x' else 'left',
#             tickfont={
#                     # 'family': 'Times',
#                     'size': fontsize
#                     },
#             ticks='outside',
#             title={
#                     'font': {
#                             # 'family': 'Times',
#                             'color': '#000000',
#                             'size': fontsize,
#                             },
#                     'text': label,
#                     },
#             type='linear',
#             zeroline=False,
#             linecolor='black',
#             linewidth=1,
#             mirror=True,
#             )
#
#
# def point_line_distance(x0, y0, x1, y1, x2, y2):
#     return np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt(
#             (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
#
#
# def calc_distances(p, start=None, end=None):
#     """
#     Parameters
#     ----------
#     p : |NDDataset|
#     """
#     dimx = p.dims[-1]
#     x = getattr(p, dimx).data
#     z = p.data
#     # ny = z.shape[0]
#     if start is None:
#         start = 0
#     if end is None:
#         end = x.size - 1
#     distances = np.zeros(z.shape)
#     for i in range(start + 1, end):
#         distances[i] = point_line_distance(x[i], z[i], x[start], z[start], x[end], z[end])
#     return distances
#
#
# def douglas_peucker(tolerance, p, mask, start, end, it):
#     distances = calc_distances(p, start, end)
#     more = distances > tolerance
#     sum = np.count_nonzero(more)
#     if sum > 0:
#         maxindex = np.argmax(distances)
#         mask[maxindex] = False
#         m1 = douglas_peucker(tolerance, p, mask, start, maxindex, it + 1)
#         m2 = douglas_peucker(tolerance, p, mask, maxindex, end, it + 1)
#         mask = np.logical_and(mask, m1)
#         mask = np.logical_and(mask, m2)
#     else:
#         for i in range(start + 1, end):
#             if distances[i] <= tolerance:
#                 mask[i] = True
#     return mask
#
#
# def simplify_douglas_peucker(tolerance, ds):
#     # start by masking all the row of data except extremity
#     new = ds.copy()
#     new.mask = np.ones_like(ds, dtype='bool')
#     new.mask[0] = False
#     new.mask[-1] = False
#     mask = douglas_peucker(tolerance, new, new.mask.copy(), 0, new.size - 1, 0)
#     new.mask[:] = mask
#     return new
#
#
# def _compress(ds, optimisation=None):
#     """
#     reduce the number of spectra to display
#     Parameters
#     ----------
#     ds: |NDDataset|
#         The dataset to simplify
#
#     Returns
#     -------
#     a list a (x,y) traces
#     """
#     sizey, sizex = ds.shape
#
#     # # find the closeness of each trace
#     # # we always keep the first and the last
#     if (optimisation == 0 and sizey < 100) or sizey < 5:
#         return ds
#
#     COUNT = {
#             0: 1000,
#             1: 500,
#             2: 150,
#             3: 30,
#             4: 5
#             }[optimisation]
#     dd = np.sum(ds, axis=1)
#     new = dd.copy()
#     if new.size > 250:
#         COUNT = COUNT / 4
#     TOLERANCE = np.max(calc_distances(dd)) / COUNT
#     new = simplify_douglas_peucker(TOLERANCE, dd)
#
#     print(np.count_nonzero(~new.mask), COUNT)
#     keep = []
#     for i, b in enumerate(new.mask):
#         if not b:
#             keep.append(ds[i])
#     return keep
#
#
# if __name__ == '__main__':
#     pass
