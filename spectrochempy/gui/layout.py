# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
This module implements the base |Dash| class.

"""

__all__ = []

import base64
import os

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_defer_js_import as dji
import dash_daq as daq


def numeric_input(id, placeholder=''):
    # we use text input as number doesn't work for float in dash-dbc
    return dbc.Input(type="text", id=id, pattern=r'[+-]{0,1}\d+.{0,1}\d*', debounce=True, placeholder=placeholder)


def card_body(children, id=None, className=None):
    if id is None:
        return dbc.CardBody(children=children, className=className)
    else:
        return dbc.CardBody(children=children, className=className, id=id)


def collapsable_card(*, body=None, header=None, key=None, id=None, className=None):
    # make a card
    return dbc.Card([dbc.CardHeader(children=header), dbc.Collapse(id=f'open-{key}-more', is_open=True,
                                                                   children=card_body(children=body, id=id,
                                                                                      className=className))])


def card(*, body=None, className=None):
    # make a card
    return dbc.Card(card_body(children=body, className=className))


def generic_card(key, title, markdown='', children=None):
    return collapsable_card(
            header=[html.H5(title), dbc.Button("More", id=f'{key}-more', outline=True, color="info", size='sm', )],
            body=[dcc.Markdown(children=markdown, className="Markdown"), html.Div(children, id=key)], key=key, )


class Layout(object):

    def __init__(self, calling_script_name='', fix_navbar=False, data_storage='session'):
        self.data_storage = data_storage
        self.STORE = [dcc.Store(id='original-data', storage_type=self.data_storage),
                      dcc.Store(id='intermediate-data', storage_type=self.data_storage),
                      dcc.Store(id='action-history', storage_type=self.data_storage)]
        self.calling_script_name = calling_script_name
        self.sticky = 'top' if not fix_navbar else None

    # ------------------------------------------------------------------------------------------------------------------
    # TOP BAR
    # ------------------------------------------------------------------------------------------------------------------
    def topbar(self):
        self.LOGO = html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(
                open(
                    os.path.join(os.path.dirname(os.path.abspath(self.calling_script_name)), 'assets', 'scpy_logo.png'),
                    'rb').read()).decode()), width='75px', )

        self.TITLE = dbc.NavbarBrand(children=dcc.Markdown('**S**pectro**C**hem**P**y *by Dash*'), id="navbarbrand",
                                     className='navbarbrand')

        return dbc.Navbar([self.LOGO, self.TITLE], color="dark", dark=True, sticky=self.sticky, )

    # ------------------------------------------------------------------------------------------------------------------
    # DATA TAB
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def about_tab():
        return card(body=dcc.Markdown("""

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
Arnaud Travert & Christian Fernandez, (2020) SpectroChemPy (Version 0.1). Zenodo. 
http://doi.org/10.5281/zenodo.3823841
```

##### Source repository 

The source are versioned using the git system and hosted on the GitHub platform: 
https://github.com/spectrochempy/spectrochempy

##### License

[CeCILL-B FREE SOFTWARE LICENSE AGREEMENT](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)


""", className='markdown',),
                    className='control-tab')

    # ------------------------------------------------------------------------------------------------------------------
    # PROJECT TAB
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def upload_form():
        return card(body=[
                dcc.Upload(id='upload-data',
                           children=dbc.Button("Upload data", outline=True, color="info", size='sm', ),
                           # Allow multiple files to be uploaded
                           multiple=True, style={
                            'float': 'left'
                            }, ),
                dbc.Button("Clear current data", color="danger", outline=True, size='sm', id='close-data'),
                ])

    def project_tab(self):
        return dbc.Card(
                dbc.CardBody(
                        [
                                dbc.InputGroup(
                                        [
                                                dbc.InputGroupAddon(
                                                        dbc.DropdownMenu(label="Project",
                                                                         bs_size='sm',
                                                                         addon_type='prepend',
                                                                         children=[
                                                                                 dbc.DropdownMenuItem("New",
                                                                                                      id='project-new'),
                                                            dbc.DropdownMenuItem(dcc.Upload("Open", id='upload-project'),
                                                                                                      id='project-open'),
                                                                                 dbc.DropdownMenuItem("Save",
                                                                                                      id='project-save',
                                                                                                      disabled=True),
                                                                                 dbc.DropdownMenuItem("Close",
                                                                                                      id='project-close',
                                                                                                      disabled=True),
                                                                                 ])
                                                        ),
                                                dbc.Input(
                                                        id='project-name',
                                                        placeholder='',
                                                        disabled=True,
                                                        size='sm')
                                                ]),
                                dbc.Collapse(id='show-project',
                                             children=[
                                                     self.upload_form(),
                                                     dbc.Collapse(id='show-project-data',
                                                                  children=generic_card(
                                                                          key='current-data',
                                                                          title='Uploaded data',
                                                                          markdown=''
                                                                          # children filled from a callback
                                                                          )
                                                                  )
                                                     ],
                                             )
                                ], style={
                                'min-height': '200px'
                                }),
                className='control-tab')

    # ------------------------------------------------------------------------------------------------------------------
    # DATA TAB
    # ------------------------------------------------------------------------------------------------------------------

    def roi_content(self):
        return [dcc.Markdown("""
Most SpectroChemPy's processing and analysis methods operate only on a region of interest (roi). 
By default, it is set to the whole range of data.

This set up is done on the ``Original`` data. The limit are automatically visualized on the graph and applied in the 
``Processed`` graph. 
If an offset is defined then the actual roi below is displayed accordingly (*see actual roi values below*).             
                                      """, className='markdown'),  # roi x limits
                dbc.InputGroup([dbc.InputGroupAddon(id='x-roi', addon_type="prepend"),
                                dbc.InputGroupAddon(id='x-roi-units', addon_type="append"), ], size='sm'),
                dbc.InputGroup([numeric_input(id='x-roi-lower', placeholder='Lower limit'),
                                numeric_input(id='x-roi-upper', placeholder="Upper limit"), ], size='sm', ),
                # x axis offset (default = 0)
                dbc.InputGroup([dbc.InputGroupAddon('offset', addon_type="prepend"),
                                numeric_input(id='x-offset', placeholder="Offset value, default=0"),

                                ], size='sm', ), dbc.FormText(id='actual-x-roi'),
                html.Hr(),
                # roi y limits
                dbc.InputGroup([dbc.InputGroupAddon(id='y-roi', addon_type="prepend"),
                                dbc.InputGroupAddon(id='y-roi-units', addon_type="append"), ], size='sm'),
                dbc.InputGroup([numeric_input(id='y-roi-lower', placeholder="Lower limit"),
                                numeric_input(id='y-roi-upper', placeholder="Upper limit"), ], size='sm', ),
                # y axis offset (default = 0)
                dbc.InputGroup([dbc.InputGroupAddon('Offset', addon_type="prepend"),
                                numeric_input(id='y-offset', placeholder="Offset value, default=0"), ], size='sm', ),
                dbc.FormText(id='actual-y-roi'),
                html.Hr(),
                ]

    def mask_content(self):
        return [dcc.Markdown("""
To mask a region, click on the **Select region button** then select your region on the graph.
To mask one of the trace, click on it (selected trace are highlihted. Confirmation will be asked before saving the masks. 
                """, className='markdown'),  # roi x limits
                dbc.InputGroup([dbc.InputGroupAddon("Masks", id='masks', addon_type="prepend"),
                                dbc.Button("Select region", color="secondary",
                                           outline=True, size='sm', id='select-mask')], size='sm'),
                html.Pre(children="No masks selected", id='text-data'),
                dcc.ConfirmDialog(
                        id='confirm-mask',
                        ),
                ]

    def data_tab(self):
        return dbc.Card(dbc.CardBody([
                dbc.Collapse(id='show-current-data',
                             children=[
                                     generic_card(
                                             key='roi',
                                             title='Region of interest',
                                             markdown="",
                                             children=[*self.roi_content()]
                                             ),
                                     generic_card(
                                             key='mask',
                                             title='Mask selection',
                                             markdown="",
                                             children=[*self.mask_content()]
                                             )
                                     ])
                ], ), className='control-tab')

    # ------------------------------------------------------------------------------------------------------------------
    # GRAPH TAB
    # ------------------------------------------------------------------------------------------------------------------

    def graph_tab(self):
        return dbc.Card(
                dbc.CardBody(
                        [generic_card('layout',
                                      title='Graph layout',
                                      markdown="",
                                      children=[
                                              dbc.Button("Zoom reset", color="secondary", outline=True, size='sm', id='zoom-reset'),
                                              dcc.Markdown("*To reset to full range display, one can also double click on the graph area (It will be faster!)* "),

                                              # data type selection
                                              dcc.Markdown("###### Data to display"),
                                              dcc.Checklist(
                                                      id='graph-selector',
                                                      options=[
                                                              {'label': 'Processed', 'value': 'Processed'},
                                                              {'label': 'Transposed', 'value': 'Transposed'},
                                                              ],
                                                      value=['Processed'],
                                                      labelStyle={'display': 'inline-block',
                                                                  'margin-right':'10px'}),

                                              #optimisation
                                                dcc.Markdown("###### Graph display optimisation"),
                                                html.Div(daq.Slider(min=0, max=4, value=3, step=1, marks={
                                                        '0'  : 'None',
                                                        '1': 'Low',
                                                        '2'  : 'Moderate',
                                                        '3': 'High',
                                                        '4'  : 'Severe'
                                                        }, id='graph-optimisation', ), style={
                                                        'margin-left'  : '30px',
                                                        'margin-bottom': '30px'
                                                        }),

                                              # Colormap
                                              dcc.Markdown("###### Colormap"),
                                              dcc.Dropdown(id='cmap-select',
                                                           options=[
                                                           {'label': 'jet', 'value': 'jet'},
                                                           {'label': 'jetr', 'value': 'jet_r'},
                                                           {'label': 'viridis', 'value': 'viridis'},
                                                           {'label': 'viridis_r', 'value': 'viridis_r'},
                                                           {'label': 'magma', 'value': 'magma'},
                                                           {'label': 'magma_r', 'value': 'magma_r'},
                                                                   ],
                                                           value='jet',
                                                           style={'width':'150px'}
                                                           )
                                              ],
                                      ),
                         generic_card('xaxis', 'Horizontal axis layout', """
                                 xaxis description
                                 """),
                         generic_card('zaxis', 'vertical axis layout', """
                                 zaxis description
                                 """),
                         ]
                        ),
                className='control-tab')

    # ------------------------------------------------------------------------------------------------------------------
    # PROCESSING TAB
    # ------------------------------------------------------------------------------------------------------------------

    def baseline(self):
        return [html.Hr(), html.H6('Method'), dcc.Dropdown(options=[{
                'label': 'Detrend',
                'value': 'detrend'
                }, {
                'label': 'PChip',
                'value': 'pchip'
                }, {
                'label': 'Polynomial',
                'value': 'polynomial'
                }, {
                'label': 'Multivariate',
                'value': 'multivariate'
                }, ], multi=False, id='subtraction_correction')
                ]

    def subtraction(self):
        return [html.Hr(),
                html.H6('Method'),
                dcc.Dropdown(options=[{
                'label': 'Subtract first row',
                'value': 'first'
                }, {
                'label': 'Subtract last row',
                'value': 'last'
                }, {
                'label': 'Subtract an external reference spectra',
                'value': 'external'
                },  ], multi=False, id='subtraction-correction'),
                html.Div(id='baseline_parameters')]

    def processing_tab(self):
        return dbc.Card(dbc.CardBody(
                [generic_card('subtraction', 'Subtraction correction',  # DO NOT FORGET TO ADD NEW KEYS IN CALLBACK
                              children=self.subtraction()),
                 generic_card('baseline', 'Baseline correction', children=self.baseline()),
                 generic_card('peakpicking', 'Peak Picking', """
                         essai
                         """),

                 ]), className='control-tab')

    # ------------------------------------------------------------------------------------------------------------------
    # TABS ZONE
    # ------------------------------------------------------------------------------------------------------------------

    def tabs_zone(self):
        return dbc.Col(
                [
                        dbc.Tabs([dbc.Tab(self.about_tab(), label="About", tab_id='about'),
                                  dbc.Tab(self.project_tab(), label="Project", tab_id='project'),
                                  dbc.Tab(self.data_tab(), label="Data", tab_id='data',
                                          id='data-tab'),
                                  dbc.Tab(self.graph_tab(), label="Graph", tab_id='graph',
                                          id='graph-tab'),
                                  dbc.Tab(self.processing_tab(), label="Processing", tab_id='processing',
                                         id='processing-tab')
                                  ], active_tab='project'),
                        ],
                       width=4, id='scpy-control-tabs',
                       className='control-tabs')

    # ------------------------------------------------------------------------------------------------------------------
    # DISPLAY ZONE
    # ------------------------------------------------------------------------------------------------------------------

    def graph_zone(self):
        graph_selector = dbc.ButtonGroup([

                                   ],)
        config = {
                'scrollZoom': True,
                'doubleClick': 'reset',
                'displayModeBar': False,
                'displaylogo': False
                }


        graph = dcc.Loading(dcc.Graph(id='graph',
                                      config=config,
                                      responsive=True, ), type='circle', )

        return dbc.Col(dbc.Collapse([
                graph,
                graph_selector,
                ],
                id='show-graph'), )

    # ------------------------------------------------------------------------------------------------------------------
    # FINAL LAYOUT
    # ------------------------------------------------------------------------------------------------------------------

    def layout(self):
        mathjax_script = dji.Import(
                src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")

        refresh_plots = dji.Import(src="https://www.spectrochempy.fr/gui/redraw.js")

        layout = html.Div([html.P(""),
                           dbc.Container(children=[dbc.Row([self.tabs_zone(), self.graph_zone()], ), ], fluid=True,
                                         style={
                                                 "font-size": '10pt',
                                                 'height'   : '80vh'
                                                 }, ), refresh_plots, mathjax_script, ], )

        return layout

    # ......................................................................................................................
    def app_page_layout(self):
        return html.Div(id='main_page',
                        children=[dcc.Location(id='url', refresh=False), *self.STORE, self.topbar(), self.layout(), ])
