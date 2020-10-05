# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
This module implements the base |Dash| class.

"""

__all__ = ['DashBoard']

import base64
import os

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_defer_js_import as dji

import spectrochempy as scp
from .plotly import Plotly

# PREFERENCES
_DATA_STORAGE_ = "session"


class DashBoard(object):

    def __init__(self, calling_script_name=''):

        self.calling_script_name = calling_script_name

        self.SCPY_LOGO = os.path.join(
            os.path.dirname(os.path.abspath(calling_script_name)), 'assets', 'scpy_logo.png'
            )

    # utilities

    @staticmethod
    def numeric_input(id, placeholder=''):
        # we use text input as number doesn't work for float in dash-dbc
        return dbc.Input(type="text", id=id, pattern='[+-]{0,1}\d+.{0,1}\d*', debounce=True, placeholder=placeholder)

    # Page design
    # -----------------------------------------------------------------------------------------------------------------

    def topbar(self):

        return dbc.Navbar(
            [
                # LOGO
                html.Img(
                    src='data:image/png;base64,{}'.format(
                        base64.b64encode(
                            open(self.SCPY_LOGO, 'rb'
                                 ).read()
                            ).decode()
                        ),
                    width='75px',
                    ),

                # APPLICATION TITLE
                dbc.NavbarBrand(
                    children='Dash-SpectroChemPy',
                    className="ml-2",
                    id="navbarbrand",
                    ),

                ],
            color="dark",
            dark=True,
            # sticky='top',
            )

    # TABS
    @staticmethod
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

    @staticmethod
    def generic_card(
            key,
            title,
            markdown='',
            children=[]
            ):

        return dbc.Card(

            dbc.CardBody(
                [
                    # title
                    html.H5(title),
                    dbc.Button("More",
                               id=f'{key}-more',
                               outline=True,
                               color="info",
                               size='sm',
                               ),
                    dbc.Collapse(
                        id=f'open-{key}-more',
                        is_open=True,
                        children=[
                            dcc.Markdown(
                                children=markdown,
                                className="Markdown"),
                            html.Div(
                                children,
                                id=key
                                )
                            ]
                        ),
                    ]
                )
            )

    @staticmethod
    def upload_form():
        return dbc.Card(dbc.CardBody(
            [
                html.H5("Input data"),
                dcc.Upload(id='upload-data',
                           children=html.Div(['Drag & Drop or ', html.A('Select file')]),
                           style={
                               'width':        '100%',
                               'height':       '30px',
                               'lineHeight':   '30px',
                               'borderWidth':  '1px',
                               'borderStyle':  'dashed',
                               'borderRadius': '5px',
                               'textAlign':    'center',
                               'margin':       '10px'
                               },
                           # Do not Allow multiple files to be uploaded
                           multiple=True)
                ]
            ))

    def ROI_content(self):

        return [
            dcc.Markdown(
                """
Most SpectroChemPy's processing and analysis methods operate only on a region of interest (ROI). 
By default, it is set to the whole range of data.

The set up is done on the original data. The limit are automatically visualized on the graph. 
If an offset is defined then the actual ROI below is displayed accordingly (*see actual ROI values below*).             
                                      """,
                className='markdown'
                ),
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
                    self.numeric_input(id='x-ROI-lower',
                                       placeholder='Lower limit'),
                    self.numeric_input(id='x-ROI-upper',
                                       placeholder="Upper limit"),
                    ],
                size='sm',
                ),
            # x axis offset (default = 0)
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon('offset',
                                        addon_type="prepend"),
                    self.numeric_input(id='x-ROI-off',
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
                    self.numeric_input(id='y-ROI-lower',
                                       placeholder="Lower limit"),
                    self.numeric_input(id='y-ROI-upper',
                                       placeholder="Upper limit"),
                    ],
                size='sm',
                ),
            # y axis offset (default = 0)
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon('Offset',
                                        addon_type="prepend"),
                    self.numeric_input(id='y-ROI-off',
                                       placeholder="Offset value, default=0"),
                    ],
                size='sm',
                ),
            dbc.FormText(id='actual-y-ROI'),
            ]

    def data_tab(self):

        return dbc.Card(
            dbc.CardBody(
                [
                    dbc.Collapse(self.upload_form(), id='show-upload-form'),
                    dbc.Collapse(id='show-current-data',
                                 children=[
                                     self.generic_card('current-data', 'Uploaded data', markdown=''),
                                     dbc.Button("Clear current data",
                                                color="danger",
                                                outline=True,
                                                size='sm',
                                                # style={'float': 'right', 'margin-top': '10px'},
                                                id='close-data'),
                                     self.generic_card(key='ROI',
                                                       title='Region of interest',
                                                       markdown="",
                                                       children=[
                                                           *self.ROI_content()
                                                           ],
                                                       ),
                                     ]
                                 ),
                    ], ),
            className='control-tab')

    def graph_tab(self):
        return dbc.Card(
            dbc.CardBody(
                [
                    self.generic_card('layout', 'Graph layout',
                                      """
                                      layout description
                                      """),
                    self.generic_card('xaxis', 'Horizontal axis layout',
                                      """
                                      xaxis description
                                      """),
                    self.generic_card('zaxis', 'vertical axis layout',
                                      """
                                      zaxis description
                                      """),
                    ]
                ),
            className='control-tab')

    def baseline(self):
        return [
            # pipeline
            html.Hr(),
            html.H6('Method'),
            dcc.Dropdown(
                options=[
                    {'label': 'Detrend', 'value': 'detrend'},
                    {'label': 'PChip', 'value': 'pchip'},
                    {'label': 'Polynomial', 'value': 'polynomial'},
                    {'label': 'Multivariate', 'value': 'multivariate'},
                    ],
                multi=False,
                id='baseline_correction'
                ),
            html.Div(id='baseline_parameters')
            ]

    def processing_tab(self):
        return dbc.Card(
            dbc.CardBody(
                [
                    self.generic_card('substraction', 'Substration'),
                    self.generic_card('baseline', 'Baseline correction',
                                      children=self.baseline()),
                    self.generic_card('peakpicking', 'Peak Picking'),
                    ]
                ),
            className='control-tab')

    # Layout
    def layout(self):

        tabs = dbc.Tabs(
            [
                dbc.Tab(self.about_tab(), label="About", tab_id='about'),
                dbc.Tab(self.data_tab(), label="Data", tab_id='data'),
                dbc.Tab(self.graph_tab(), label="Graph", tab_id='graph', id='graph-tab'),
                dbc.Tab(self.processing_tab(), label="Processing", tab_id='processing', id='processing-tab')
                ],
            active_tab='data',
            )

        graph_tabs = dbc.Tabs(
            [

                dbc.Tab(
                    dcc.Loading(
                        dcc.Graph(
                            id='graph',
                            responsive=True,
                            ),
                        type='circle',
                        ),
                    id='tab_original',
                    label='Original',
                    tab_id='original'),

                
                dbc.Tab(
                    id='tab_processed',
                    disabled=True),
                ],

            active_tab='original',
            )

        mathjax_script = dji.Import(
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")
        refresh_plots = dji.Import(src="https://www.spectrochempy.fr/gui/redraw.js")

        layout = html.Div([
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
                            dbc.Collapse(
                                html.Div(
                                    graph_tabs,
                                    id='scpy-graph-tabs',
                                    className='control-tabs'),
                                id='show-graph'),
                            )
                        ],
                    ),
                    # data storage
                    dcc.Store(id='original-data', storage_type=_DATA_STORAGE_),
                    dcc.Store(id='intermediate-data', storage_type=_DATA_STORAGE_),
                    dcc.Store(id='processing_history', storage_type=_DATA_STORAGE_)

                    ],
                fluid=True,
                style={"font-size": '10pt', 'height': '80vh'},
                ),
            refresh_plots,
            mathjax_script,
            ], )

        return layout

    #######################################################################################################################
    # Callbacks and related function
    #######################################################################################################################

    @staticmethod
    def parse_upload_contents(filename, contents):
        #
        # transform uploaded content to a NDDataset
        #
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        ds = scp.NDDataset.read(filename, content=decoded)
        return ds.write_json(to_string=True)

    @staticmethod
    def dataset_itemlist(ds):
        return [
            html.H6(f'{ds.name}'),
            html.Code(f'{str(ds)}'),
            # dcc.Markdown(f'{ds.description}'),
            # *[dcc.Markdown(f'{d}') for d in ds.history],
            ]

    def dataset_list(self, *datasets):
        return [dbc.ListGroupItem([
            dbc.Checkbox(id=f"dataset-checkbox-{i}", style={'float': 'right'}, checked=True),
            *self.dataset_itemlist(ds)]) for i, ds in enumerate(datasets)]

    def uploaded_dataset_list(self, *datasets):
        #
        # TODO: active can be set to True for highligthing selected File (however for the moment we limit to one component)
        # TODO: write  a callback for each entry
        #
        all_datasets = [
            dbc.ListGroup(self.dataset_list(*datasets))
            ]
        return all_datasets

    ############
    # CALLBACK #
    ############

    def callbacks(self, app):
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
                data = {n: self.parse_upload_contents(n, c) for n, c in z}
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
                      [State('original-data', 'data'),
                       State('intermediate-data', 'data')])
        def save_intermediate_data(xrl, xru, yrl, yru, xrf, yrf, ts, data, saved):
            if data is None:
                raise PreventUpdate

            # .... make oll the transformation
            datasets = [scp.read_json(content=c.encode()) for n, c in data.items()]
            if saved:
                saved_datasets = [scp.read_json(content=c.encode()) for n, c in saved.items()]
                diff = []
                for ds, sds in zip(datasets, saved_datasets):
                    diff.append(ds != sds)

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
            newdata = {ds.filename: ds.write_json(to_string=True) for ds in datasets}

            return newdata, actx, acty

        # -----------------------
        # UPDATE DATA TAB CONTENT
        # -----------------------
        @app.callback([Output('current-data', 'children'),
                       Output('show-upload-form', 'is_open'),
                       Output('show-current-data', 'style'),
                       Output('show-graph', 'is_open'),
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

            dataloaded = None
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
                dataloaded = self.uploaded_dataset_list(*datasets)
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

            return (dataloaded, is_open, style, not is_open,
                    xr, yr, xrl, xru, yrl, yru,
                    xr_units, yr_units, xro, yro)

        # @app.callback(Output('show_processed', ''),
        #               [  # ROI
        #                   Input('x-ROI-lower', 'value'),
        #                   Input('x-ROI-upper', 'value'),
        #                   Input('y-ROI-lower', 'value'),
        #                   Input('y-ROI-upper', 'value'),
        #                   Input('x-ROI-off', 'value'),
        #                   Input('y-ROI-off', 'value')],
        #               [Input('original-data', 'modified_timestamp')],
        #               [State('original-data', 'data'),
        #                State('intermediate-data', 'data')])
        # def save_intermediate_data(xrl, xru, yrl, yru, xrf, yrf, ts, data, saved):
        #     if data is None:
        #         raise PreventUpdate
        #
        #     # .... make oll the transformation
        #     datasets = [scp.read_json(content=c.encode()) for n, c in data.items()]
        #     if saved:
        #         saved_datasets = [scp.read_json(content=c.encode()) for n, c in saved.items()]
        #         diff = []
        #         for ds, sds in zip(datasets, saved_datasets):
        #             diff.append(ds!=sds)

        # -------------
        # UPDATE FIGURE
        # -------------
        @app.callback(Output('graph', 'figure'),
                      Input('intermediate-data', 'data'), )
        def update_figure(data):

            if data is None:
                raise PreventUpdate

            datasets = [scp.read_json(content=c.encode()) for n, c in data.items()]

            ds = datasets[0]
            figure = Plotly.plot_stack(ds)

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
            [Input('current-data', 'children')])
        def tab_display_control(children):
            if not children:
                return True, True
            else:
                return False, False

        # ----------------------------------
        # MODIFY CLOSING/OPENING CARD BUTTON
        # ----------------------------------
        for item in ['ROI', 'current-data',
                     'layout', 'xaxis', 'zaxis',
                     'baseline', 'peakpicking']:
            @app.callback(
                [Output(f"open-{item}-more", "is_open"),
                 Output(f"{item}-more", "children"),
                 Output(f"{item}-more", "color")],
                [Input(f"{item}-more", "n_clicks")],
                [State(f"{item}-more", "children")]
                )
            def on_click(n, state):
                if n is None or state.startswith('Close'):
                    return False, "More", "info"
                else:
                    return True, "Close this card", "warning"

    # ......................................................................................................................
    def app_page_layout(self, **kwargs):

        return html.Div(
            id='main_page',
            children=[
                dcc.Location(id='url', refresh=False),
                self.topbar(**kwargs),
                html.Div(
                    id='app-page-content',
                    children=self.layout()),
                ],
            )
