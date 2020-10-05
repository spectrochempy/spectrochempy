# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""define plots"""

import plotly.graph_objects as go

from spectrochempy.utils.colors import colorscale

__all__ = ['Plotly']


class Plotly(object):

    @classmethod
    def plot_stack(cls, dataset=None, **kwargs):
        """
        Generate a Plotly stacked plot of a 2D dataset

        Parameters
        ----------
        dataset; |NDDataset|
            The dataset to plot
        kwargs: any
            Additional keyword arguments

        Returns
        -------
        figure
            A plotly figure

        """
        if dataset is None:
            return []

        # Here I am using matplotlib to find the color map
        # because I did not find the way to do this easily using plotly)
        cmap = 'viridis'
        colorscale.normalize(dataset.y.data, cmap=cmap)

        data = []
        x = dataset.x - dataset.x.offset_value
        for i, row in enumerate(dataset):
            trace = row.data.squeeze()
            name = f'{row.y.values - dataset.y.offset_value:~P}'
            data.append(dict(x=x.data,
                             xaxis='x',
                             y=trace,
                             yaxis='y',
                             name=name,
                             mode='lines',
                             type='scattergl',
                             line=dict(
                                 color=colorscale.rgba(row.y.data),
                                 dash='solid',
                                 width=1.0),
                             )
                        )

        # Layout
        roirange = dataset.x.roi.copy()
        fullrange = dataset.x.limits.copy()

        if dataset.x.reversed:
            roirange.reverse()
            fullrange.reverse()

        layout = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(220,220,220,1)',
            autosize=True,
            # height=kwargs.get('height', 800),
            hovermode='closest',
            showlegend=False,
            )

        def make_axis(axis,
                      range=None,
                      label=None,
                      **kwargs):

            fontsize = kwargs.get('fontsize', 18)

            return dict(
                anchor='y' if axis == 'x' else 'x',
                domain=[0.0, 1.0],
                nticks=7,
                range=range,
                showgrid=True,
                side='bottom' if axis == 'x' else 'left',
                tickfont={  # 'family': 'Times',
                    'size': fontsize
                    },
                ticks='outside',
                title={
                    'font': {
                        # 'family': 'Times',
                        'color': '#000000',
                        'size':  fontsize,
                        },
                    'text': label,
                    },
                type='linear',
                zeroline=False,
                linecolor='black',
                linewidth=1,
                mirror=True,
                )

        layout.update(
            dict(
                xaxis=make_axis('x', range=roirange, label=f'{dataset.x.title} / {dataset.x.units:~P}', **kwargs),
                yaxis=make_axis('y', label=f'{dataset.title} / {dataset.units:~P}', **kwargs),
                )
            )

        # Create traces

        fig = go.Figure(
            data=data,
            layout=go.Layout(layout),
            )

        # show out of ROI zone
        x0, x1 = fullrange[0], roirange[0]
        x2, x3 = fullrange[1], roirange[1]
        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    # x-reference is assigned to the x-values
                    xref="x",
                    # y-reference is assigned to the plot paper [0,1]
                    yref="paper",
                    x0=x0,
                    y0=0,
                    x1=x1,
                    y1=1,
                    fillcolor="LightSalmon",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    ),
                dict(
                    type="rect",
                    # x-reference is assigned to the x-values
                    xref="x",
                    # y-reference is assigned to the plot paper [0,1]
                    yref="paper",
                    x0=x2,
                    y0=0,
                    x1=x3,
                    y1=1,
                    fillcolor="LightSalmon",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    ),
                ]
            )
        return fig

Plotly = Plotly()