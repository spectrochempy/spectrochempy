# -*- coding: utf-8 -*
# Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                        

# SpectroChemPy GUI: Dash Application
# **Note:** Run this app with the command `scpy` in a terminal and then visit http://127.0.0.1:8050/ in your web
# browser.

# TODO: By setting multiple=True in the Upload components we could load several file (but we need to handle this)

import base64
import os
import webbrowser
from threading import Timer

# from jupyter_dash import JupyterDash
from dash import Dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
# see https://dash-bootstrap-components.opensource.faculty.ai/docs
import dash_defer_js_import as dji

import spectrochempy as scp
from spectrochempy.utils.colors import colorscale

# PREFERENCES
_DATA_STORAGE_ = "session"


# Page design
# ---------------------------------------------------------------------------------------------------------------------

# NAVBAR
def logo(filename):
    SCPY_LOGO = os.path.join(os.path.dirname(os.path.abspath(filename)), 'assets', 'scpy_logo.png')
    return html.Img(
            src='data:image/png;base64,{}'.format(
                    base64.b64encode(
                            open(SCPY_LOGO, 'rb'
                                 ).read()
                            ).decode()
                    ),
            width='75px',
            )


def navbarbrand():
    return dbc.NavbarBrand(
            children='SpectroChemPy',
            className="ml-2",
            id="navbarbrand",
            )


def doc():
    return dbc.NavItem(
            dbc.NavLink(
                    "Documentation",
                    href="https://www.spectrochempy.fr",
                    target="_blank",
                    ),
            )


def nav_menus():
    dropdown = dbc.DropdownMenu(
            children=[dbc.DropdownMenuItem("Open", id='menu_open'), dbc.DropdownMenuItem("Close", id='menu_close'),
                      dbc.DropdownMenuItem("Save", id='menu_save'), dbc.DropdownMenuItem(divider=True),
                      dbc.DropdownMenuItem("Preferences", id='menu_preferences'), ], nav=True, in_navbar=True,
            label="File", )
    return [dropdown]


def navbar(filename, **kwargs):
    _menus = kwargs.get('menus', None)
    menus = _menus() if _menus is not None else []
    menus.append(doc())
    nav = dbc.Collapse(
            dbc.Nav(
                    children=menus,
                    fill=True,
                    ),
            id="navbar-collapse",
            navbar=True,
            )

    return dbc.Navbar(
            [
                    logo(filename),
                    navbarbrand(),

                    # NO vavigation bar

                    # dbc.NavbarToggler(
                    # 	 id="navbar-toggler",
                    # ),
                    # nav,
                    ],
            color="dark",
            dark=True,
            # sticky='top',
            )


# TABS
def about_tab():
    return dbc.Card(dbc.CardBody([

            dcc.Markdown("""

##### What is SpectroChemPy?

**SpectroChemPy** is a framework for processing, analyzing and modeling Spectroscopic 
data for Chemistry with Python. It is a cross platform software, running on Linux, 
Windows or OS X.

##### Documentation

The online Html documentation is available here:  [HTML documentation](https://www.spectrochempy.fr)

##### Issue Tracker

SpectroChemPy is still experimental and under active development. 
Its current design is subject to major changes, reorganizations, bugs and crashes!!!. 

You find a problem, want to suggest enhancements or want to look at the current issues and milestones, 
you can go there:  [Issue Tracker](https://github.com/spectrochempy/spectrochempy/issues)

##### Citing SpectroChemPy

When using SpectroChemPy for your own work, you are kindly requested to cite it this way:
```text
Arnaud Travert & Christian Fernandez, (2020) SpectroChemPy (Version 0.1). Zenodo. http://doi.org/10.5281/zenodo.3823841
```

##### Source repository 

The source are versioned using the git system and hosted on the GitHub platform: 
https://github.com/spectrochempy/spectrochempy

##### License

[CeCILL-B FREE SOFTWARE LICENSE AGREEMENT](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)


""", className='markdown'), ]),
            className='control-tab')


def numeric_input(id, placeholder=''):
    # we use text input as number doesn't work for float in dash-dbc
    return dbc.Input(type="text", id=id, pattern='[+-]{0,1}\d+.{0,1}\d*', debounce=True, placeholder=placeholder)


def data_tab_content():
    content = [
            dbc.Card(

                    # ROI
                    # -----------------------------------------------------------------------------------------------------
                    dbc.CardBody(
                            [
                                    # title
                                    html.H5('Region of Interest (ROI)', id='roi'),
                                    dbc.Button("Modify", id='ROI-modify', outline=True, color="info",
                                               size='sm', className="mr-1"),
                                    dbc.Collapse(
                                            id='open-ROI-modify',
                                            children=[
                                                    dcc.Markdown(
                                                            """
                                        Most SpectroChemPy's processing and analysis methods operate only on a region of interest (ROI). 
                                        By default, it is set to whole range of data.
                                        
                                        The set up is done on the original data. The limit are automatically visualized on the graph. 
                                        If an offset is defined then the actual ROI below is displayed accordingly (*see actual ROI values below*).             
                                                            """,
                                                            className="Markdown"),

                                                    # ROI x limits
                                                    dbc.InputGroup(
                                                            [
                                                                    dbc.InputGroupAddon(id='x-ROI',
                                                                                        addon_type="prepend"),
                                                                    dbc.InputGroupAddon(id='x-ROI-units',
                                                                                        addon_type="append"),
                                                                    ],
                                                            size='sm'),
                                                    dbc.InputGroup(
                                                            [
                                                                    numeric_input(id='x-ROI-lower',
                                                                                  placeholder='Lower limit'),
                                                                    numeric_input(id='x-ROI-upper',
                                                                                  placeholder="Upper limit"),
                                                                    ],
                                                            size='sm',
                                                            ),
                                                    # x axis offset (default = 0)
                                                    dbc.InputGroup(
                                                            [
                                                                    dbc.InputGroupAddon('offset',
                                                                                        addon_type="prepend"),
                                                                    numeric_input(id='x-ROI-off',
                                                                                  placeholder="Offset value, default=0"),

                                                                    ],
                                                            size='sm',
                                                            ),
                                                    dbc.FormText(id='actual-x-ROI'),
                                                    html.Hr(),
                                                    # ROI y limits
                                                    dbc.InputGroup(
                                                            [
                                                                    dbc.InputGroupAddon(id='y-ROI',
                                                                                        addon_type="prepend"),
                                                                    dbc.InputGroupAddon(id='y-ROI-units',
                                                                                        addon_type="append"),
                                                                    ],
                                                            size='sm'),
                                                    dbc.InputGroup(
                                                            [
                                                                    numeric_input(id='y-ROI-lower',
                                                                                  placeholder="Lower limit"),
                                                                    numeric_input(id='y-ROI-upper',
                                                                                  placeholder="Upper limit"),
                                                                    ],
                                                            size='sm',
                                                            ),
                                                    # y axis offset (default = 0)
                                                    dbc.InputGroup(
                                                            [
                                                                    dbc.InputGroupAddon('Offset',
                                                                                        addon_type="prepend"),
                                                                    numeric_input(id='y-ROI-off',
                                                                                  placeholder="Offset value, default=0"),
                                                                    ],
                                                            size='sm',
                                                            ),
                                                    dbc.FormText(id='actual-y-ROI'),
                                                    ]
                                            ),
                                    ]

                            )),
            ]
    return content


def data_tab():
    upload = dcc.Upload(id='upload-data',
                        children=html.Div(['Drag & Drop or ', html.A('Select file')]),
                        style={
                                'width'       : '100%',
                                'height'      : '30px',
                                'lineHeight'  : '30px',
                                'borderWidth' : '1px',
                                'borderStyle' : 'dashed',
                                'borderRadius': '5px',
                                'textAlign'   : 'center',
                                'margin'      : '10px'
                                },
                        # Do not Allow multiple files to be uploaded
                        multiple=True)

    upload_form = dbc.Card(dbc.CardBody(
            [
                    html.H5("Input data"),
                    upload,
                    ]
            ))

    return dbc.Card(
            dbc.CardBody(
                    [
                            dbc.Collapse(upload_form, id='show-upload-form'),
                            dbc.Collapse(
                                    [
                                            dbc.Card(
                                                    dbc.CardBody(
                                                            [
                                                                    html.Div(id='current-data'),
                                                                    dbc.Button("Clear Data",
                                                                               color="warning",
                                                                               outline=True,
                                                                               size='sm',
                                                                               # style={'float': 'right', 'margin-top': '10px'},
                                                                               id='close-data')
                                                                    ]
                                                            )
                                                    ),
                                            dbc.Collapse(data_tab_content(), id='show-close-data'),
                                            ],
                                    id='show-current-data')
                            ],
                    ),
            className='control-tab')


def graph_tab_content():
    content = [
            dbc.Card(

                    # LAYOUT
                    # -----------------------------------------------------------------------------------------------------
                    dbc.CardBody(
                            [
                                    # title
                                    html.H5('Graph layout', id='layout'),
                                    dbc.Button("Modify", id='layout-modify', outline=True, color="info",
                                               size='sm', className="mr-1"),
                                    dbc.Collapse(
                                            id='open-layout-modify',
                                            children=[
                                                    dcc.Markdown(
                                                            """
layout             
                                                            """,
                                                            className="Markdown"),
                                                    ]
                                            ),
                                    ]
                            )
                    ),
            dbc.Card(

                    # AXIS X
                    # -----------------------------------------------------------------------------------------------------
                    dbc.CardBody(
                            [
                                    # title
                                    html.H5('Horizontal axis layout', id='xaxis'),
                                    dbc.Button("Modify", id='xaxis-modify', outline=True, color="info",
                                               size='sm', className="mr-1"),
                                    dbc.Collapse(
                                            id='open-xaxis-modify',
                                            children=[
                                                    dcc.Markdown(
                                                            """
layout             
                                                            """,
                                                            className="Markdown"),
                                                    ]
                                            ),
                                    ]
                            )
                    ),
            dbc.Card(

                    # AXIS Y
                    # -----------------------------------------------------------------------------------------------------
                    dbc.CardBody(
                            [
                                    # title
                                    html.H5('Vertical axis layout', id='yaxis'),
                                    dbc.Button("Modify", id='yaxis-modify', outline=True, color="info",
                                               size='sm', className="mr-1"),
                                    dbc.Collapse(
                                            id='open-yaxis-modify',
                                            children=[
                                                    dcc.Markdown(
                                                            """
layout             
                                                            """,
                                                            className="Markdown"),
                                                    ]
                                            ),
                                    ]
                            )
                    )
        ]


    return content


def graph_tab():

    return dbc.Card(
            dbc.CardBody(
                    graph_tab_content()
                    ),
        className='control-tab')


def processing_tab_content():
    content = [
            dbc.Card(

                    # BASELINE
                    # -----------------------------------------------------------------------------------------------------
                    dbc.CardBody(
                            [
                                    # title
                                    html.H5('Baseline processing', id='baseline'),
                                    dbc.Button("Modify", id='baseline-modify', outline=True, color="info",
                                               size='sm', className="mr-1"),
                                    dbc.Collapse(
                                            id='open-baseline-modify',
                                            children=[
                                                    dcc.Markdown(
                                                            """
                                        Baseline processing             
                                                            """,
                                                            className="Markdown"),
                                                    ]
                                            ),
                                    ]
                            )
                    ),
            dbc.Card(

                    # PEAK PICKING
                    # -----------------------------------------------------------------------------------------------------
                    dbc.CardBody(
                            [
                                    # title
                                    html.H5('Peak picking', id='peakpicking'),
                                    dbc.Button("Modify", id='peakpicking-modify', outline=True, color="info",
                                               size='sm', className="mr-1"),
                                    dbc.Collapse(
                                            id='open-peakpicking-modify',
                                            children=[
                                                    dcc.Markdown(
                                                            """
                                        Peak picking             
                                                            """,
                                                            className="Markdown"),
                                                    ]
                                            ),
                                    ]
                            )
                    )
            ]

    return content

def processing_tab():
    return dbc.Card(
            dbc.CardBody(
                    processing_tab_content()
                    ),
            className='control-tab')


# Layout
def layout():
    tabs = dbc.Tabs(
            [dbc.Tab(about_tab(), label="About", tab_id='about'),
             dbc.Tab(data_tab(), label="Data", tab_id='data'),
             dbc.Tab(graph_tab(), label="Graph", tab_id='graph', id='graph-tab'),
             dbc.Tab(processing_tab(), label="Processing", tab_id='processing', id='processing-tab')
             # disabled=True),
             ],
            active_tab='data', )

    mathjax_script = dji.Import(
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")
    refresh_plots = dji.Import(src="https://www.spectrochempy.fr/gui/redraw.js")

    return html.Div([
            html.P(""),
            dbc.Container(
                    children=[dbc.Row(  # 12 columns
                            [dbc.Col(
                                    tabs,
                                    width=4,
                                    id='scpy-control-tabs',
                                    className='control-tabs'
                                    ),
                                    dbc.Col(
                                            dcc.Loading(
                                                    dcc.Graph(
                                                            id='graph',
                                                            responsive=True,
                                                            ),
                                                    type='circle',
                                                    ),
                                            width=8,

                                            ),
                                    ],
                            ),

                            # data storage
                            dcc.Store(id='original-data', storage_type=_DATA_STORAGE_),
                            dcc.Store(id='intermediate-data', storage_type=_DATA_STORAGE_),
                            ],
                    fluid=True,
                    style={"font-size": '10pt', 'height': '80vh'},
                    ),
            refresh_plots,
            mathjax_script,
            ], )

# --------------
# CREATE A PLOT
# --------------
def plotly_2D(ds=None, **kwargs):
    if ds is None:
        return []

    # use matplotlib to find the color map (did not find the way to do this using plotly)
    cmap = 'viridis'

    colorscale.normalize(ds.y.data, cmap=cmap)

    data = []
    x = ds.x - ds.x.offset_value
    for i, row in enumerate(ds):
        trace = row.data.squeeze()
        name = f'{row.y.values - ds.y.offset_value:~P}'
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
    roirange = ds.x.roi.copy()
    fullrange = ds.x.limits.copy()

    if ds.x.reversed:
        roirange.reverse()
        fullrange.reverse()

    layout = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(220,220,220,1)',
            autosize=True,
            #height=kwargs.get('height', 800),
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
                                'size' : fontsize,
                                },
                        'text': label,
                        },
                type='linear',
                zeroline=False,
                )

    layout.update(
            dict(
                    xaxis=make_axis('x', range=roirange, label=f'{ds.x.title} / {ds.x.units:~P}', **kwargs),
                    yaxis=make_axis('y', label=f'{ds.title} / {ds.units:~P}', **kwargs),
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

#######################################################################################################################
# Callbacks and related function
#######################################################################################################################


def parse_upload_contents(filename, contents):
    #
    # transform uploaded content to a NDDataset
    #
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    ds = scp.NDDataset.read(filename, content=decoded)
    return ds.write_json(to_string=True)


def dataset_itemlist(ds):
    return [
            html.H6(f'{ds.name}'),
            html.Code(f'{str(ds)}'),
            # dcc.Markdown(f'{ds.description}'),
            # *[dcc.Markdown(f'{d}') for d in ds.history],
            ]


def dataset_list(*datasets):
    return [dbc.ListGroupItem([
            dbc.Checkbox(id=f"dataset-checkbox-{i}", style={'float': 'right'}, checked=True),
            *dataset_itemlist(ds)]) for i, ds in enumerate(datasets)]


def uploaded_dataset_list(*datasets):
    #
    # TODO: active can be set to True for highligthing selected File (however for the moment we limit to one component)
    # TODO: write  a callback for each entry
    #
    all_datasets = [
            html.H5('Uploaded dataset:'),
            dbc.ListGroup(dataset_list(*datasets))
            ]
    return all_datasets


############
# CALLBACK #
############

def callbacks(app):
    """ Add all application calbacks """

    # ------------
    # DATA UPLOAD
    # ------------
    @app.callback(Output('original-data', 'data'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'))
    def data_upload(list_of_contents, list_of_names):
        # Store data in components

        data = {}  # the data must be JSON serializable

        if list_of_contents is not None:
            # data uploaded
            z = zip(list_of_names, list_of_contents)
            data = {n: parse_upload_contents(n, c) for n, c in z}
        else:
            raise PreventUpdate
        return data

    # ----------------------
    # SAVE INTERMEDIATE DATA
    # ----------------------
    @app.callback([Output('intermediate-data', 'data'),
                   Output('actual-x-ROI', 'children'),
                   Output('actual-y-ROI', 'children')],
                  [  # ROI
                          Input('x-ROI-lower', 'value'),
                          Input('x-ROI-upper', 'value'),
                          Input('y-ROI-lower', 'value'),
                          Input('y-ROI-upper', 'value'),
                          Input('x-ROI-off', 'value'),
                          Input('y-ROI-off', 'value')],
                  [Input('original-data', 'modified_timestamp')],
                  [State('original-data', 'data')])
    def save_intermediate_data(xrl, xru, yrl, yru, xrf, yrf, ts, data):
        if data is None:
            raise PreventUpdate

        # .... make oll the transformation
        datasets = [scp.read_json(content=c.encode()) for n, c in data.items()]

        # for now we work with only one spectrum
        ds = datasets[0]

        ds.x.offset = float(xrf) if xrf else 0
        ds.y.offset = float(yrf) if yrf else 0
        ds.x.roi = [float(xrl) if xrl else 0, float(xru) if xru else 0]
        ds.y.roi = [float(yrl) if yrl else 0, float(yru) if yru else 0]

        actx = f'Actual ROI: {ds.x.roi_values[0]:~P} -> {ds.x.roi_values[1]:~P}'
        acty = f'Actual ROI: {ds.y.roi_values[0]:~P} -> {ds.y.roi_values[1]:~P}'

        # create output
        datasets[0] = ds
        data = {ds.filename: ds.write_json(to_string=True) for ds in datasets}

        return data, actx, acty

    # -----------------------
    # UPDATE DATA TAB CONTENT
    # -----------------------
    @app.callback([Output('current-data', 'children'),
                   Output('show-upload-form', 'is_open'),
                   Output('show-current-data', 'is_open'),
                   Output('show-close-data', 'is_open'),
                   Output('graph', 'style'),
                   Output('x-ROI', 'children'),
                   Output('y-ROI', 'children'),
                   Output('x-ROI-lower', 'value'),
                   Output('x-ROI-upper', 'value'),
                   Output('y-ROI-lower', 'value'),
                   Output('y-ROI-upper', 'value'),
                   Output('x-ROI-units', 'children'),
                   Output('y-ROI-units', 'children'),
                   Output('x-ROI-off', 'value'),
                   Output('y-ROI-off', 'value')],
                  [Input('original-data', 'modified_timestamp')],
                  [State('original-data', 'data'),
                   State('intermediate-data', 'data')])
    def update_tab_content(ts, data, saveddata):
        if ts is None:
            raise PreventUpdate

        thumbnails = None
        is_open = True
        xr = 'x'
        yr = 'y'
        xrl, xru = None, None
        yrl, yru = None, None
        xr_units = None
        yr_units = None
        xro = 0
        yro = 0
        style = dict({'display': 'none'})

        if saveddata is not None:  # take the saved data!
            data = saveddata  # json.loads(saveddata)

        if data is not None:
            datasets = [scp.read_json(content=c.encode()) for n, c in data.items()]

            # for now we work with only one spectrum
            ds = datasets[0]
            thumbnails = uploaded_dataset_list(*datasets)
            is_open = False
            xr = ds.x.title  # we assume homogeneous data (same kind of dimension)
            yr = ds.y.title
            xrl, xru = ds.x.roi
            yrl, yru = ds.y.roi
            xr_units = f'{ds.x.units:~P}'
            yr_units = f'{ds.y.units:~P}'
            xro = ds.x.offset
            yro = ds.y.offset
            style = dict({'display': 'block'})

        data_is_open = not is_open
        button_is_open = not is_open
        return thumbnails, is_open, data_is_open, button_is_open, style, xr, yr, xrl, xru, yrl, yru, \
               xr_units, yr_units, xro, yro

    # -------------
    # UPDATE FIGURE
    # -------------
    @app.callback(Output('graph', 'figure'),
                  [Input('intermediate-data', 'data')])
    def update_figure(data):

        if data is None:
            raise PreventUpdate

        figure = go.Figure(data=[])

        if data is not None:
            datasets = [scp.read_json(content=c.encode()) for n, c in data.items()]

            ds = datasets[0]
            figure = plotly_2D(ds)

        return figure

    # ------------------
    # CLEAR DATA BUTTON
    # ------------------
    @app.callback(
            [Output("original-data", "clear_data"),
             Output("intermediate-data", "clear_data")],
            [Input("close-data", "n_clicks")]
            )
    def clear_data_click(n):
        if n is None:
            return False, False
        else:
            return True, True

    # -------------------
    # TAB DISPLAY CONTROL
    # -------------------
    @app.callback(
            [Output('graph-tab', 'disabled'),
             Output('processing-tab', 'disabled')],
            [Input('show-close-data', 'is_open')])
    def tab_display_control(is_open):
        if not is_open:
            return True, True
        else:
            return False, False

    # -----------------
    # MODIFY ROI BUTTON
    # ------------------
    @app.callback(
            [Output("open-ROI-modify", "is_open"),
             Output('ROI-modify', 'children')],
            [Input("ROI-modify", "n_clicks")],
            [State("ROI-modify", "children")]
            )
    def ROI_modify_click(n, state):
        if n is None or state == 'Close':
            return False, "Modify"
        else:
            return True, 'Close'

    # -----------------
    # MODIFY LAYOUT BUTTON
    # ------------------
    @app.callback(
            [Output("open-layout-modify", "is_open"),
             Output('layout-modify', 'children')],
            [Input("layout-modify", "n_clicks")],
            [State("layout-modify", "children")]
            )
    def graph_layout_modify_click(n, state):
        if n is None or state == 'Close':
            return False, "Modify"
        else:
            return True, 'Close'

    # -----------------
    # MODIFY XAXIS LAYOUT BUTTON
    # ------------------
    @app.callback(
            [Output("open-xaxis-modify", "is_open"),
             Output('xaxis-modify', 'children')],
            [Input("xaxis-modify", "n_clicks")],
            [State("xaxis-modify", "children")]
            )
    def xaxis_modify_click(n, state):
        if n is None or state == 'Close':
            return False, "Modify"
        else:
            return True, 'Close'

    # -----------------
    # MODIFY YAXIS LAYOUT BUTTON
    # ------------------
    @app.callback(
            [Output("open-yaxis-modify", "is_open"),
             Output('yaxis-modify', 'children')],
            [Input("yaxis-modify", "n_clicks")],
            [State("yaxis-modify", "children")]
            )
    def yaxis_modify_click(n, state):
        if n is None or state == 'Close':
            return False, "Modify"
        else:
            return True, 'Close'

    # -----------------
    # MODIFY BASELINE BUTTON
    # ------------------
    @app.callback(
            [Output("open-baseline-modify", "is_open"),
             Output('baseline-modify', 'children')],
            [Input("baseline-modify", "n_clicks")],
            [State("baseline-modify", "children")]
            )
    def baseline_modify_click(n, state):
        if n is None or state == 'Close':
            return False, "Modify"
        else:
            return True, 'Close'

    # -----------------
    # MODIFY PEAK PICKING BUTTON
    # ------------------
    @app.callback(
            [Output("open-peakpicking-modify", "is_open"),
             Output('peakpicking-modify', 'children')],
            [Input("peakpicking-modify", "n_clicks")],
            [State("peakpicking-modify", "children")]
            )
    def baseline_modify_click(n, state):
        if n is None or state == 'Close':
            return False, "Modify"
        else:
            return True, 'Close'

# ......................................................................................................................
def app_page_layout(page_layout,
                    filename,
                    **kwargs):
    return html.Div(
            id='main_page',
            children=[
                    dcc.Location(id='url', refresh=False),
                    navbar(filename, **kwargs),
                    html.Div(
                            id='app-page-content',
                            children=page_layout),
                    ],
            )


# .............................................................................
def run_standalone_app(
        layout,
        callbacks,
        filename,
        **kwargs
        ):
    """
    Run a dash application as standalone (or in Jupyter Lab)

    Parameters
    ----------
    layout: callable
    callbacks: callable
    filename: name of the calling file
    kwargs: other optional parameters

    Returns
    -------
    app: Dash application

    """

    # Initialisation............................................................

    theme = kwargs.pop('theme', dbc.themes.BOOTSTRAP)
    # CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN, LUX, MATERIA, MINTY, PULSE, SANDSTONE,
    # SIMPLEX, SKETCHY, SLATE, SOLAR, SPACELAB, SUPERHERO, UNITED, YETI

    app = Dash(__name__, title='Dash-SpectroChemPy', external_stylesheets=[theme])
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True
    app.index_string = """
	<!DOCTYPE html>
	<html>
		<head>
			{%metas%}
			<title>{%title%}</title>
			{%favicon%}
			{%css%}
		</head>
		<body>
			{%app_entry%}
			<footer>
				{%config%}
				{%scripts%}
				<script type="text/x-mathjax-config">
				MathJax.Hub.Config({
					tex2jax: {
					inlineMath: [ ['$','$'],],
					processEscapes: true
					}
				});
				</script>
				{%renderer%}
			</footer>
		</body>
	</html>
	"""

    # Assign layout
    app.layout = app_page_layout(
            page_layout=layout(),
            filename=filename,
            **kwargs,
            )

    # Register all callbacks
    callbacks(app)

    # return app object
    return app


def open_browser():
    webbrowser.open_new('http://127.0.0.1:8049/')


def main():
    app = run_standalone_app(layout, callbacks, __file__, menus=nav_menus)
    server = app.server

    Timer(1, open_browser).start()
    app.run_server(debug=True, port=8049)


if __name__ == '__main__':
    main()
