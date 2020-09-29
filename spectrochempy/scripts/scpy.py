# -*- coding: utf-8 -*
# %% [markdown]
# # SpectroChemPy GUI: Dash Application

# %% [markdown]
# Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                        

# %% [markdown]
# **Note:** Run this app with the command `scpy` in a terminal and then visit http://127.0.0.1:8050/ in your web browser.

# %%
import datetime
import base64
import io
import os

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc
# see https://dash-bootstrap-components.opensource.faculty.ai/docs

import plotly.express as px

import spectrochempy as scp
from spectrochempy.dash_gui import run_standalone_app
from spectrochempy.utils import convert_to_html
from matplotlib.figure import Figure

# .............................................................................
def menus():
    dropdown = dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Open", id='menu_open'),
                dbc.DropdownMenuItem("Close", id='menu_close'),
                dbc.DropdownMenuItem("Save", id='menu_save'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Preferences", id='menu_preferences'),
            ],
            nav=True,
            in_navbar=True,
            label="File",
    )
    return [dropdown]

# .............................................................................
def about_tab():
    return dbc.Card(
            dbc.CardBody(
                    [

                        dcc.Markdown(
"""
### What is 

SpectroChemPy is a framework for processing, analyzing and modeling Spectroscopic 
data for Chemistry with Python. It is a cross platform software, running on Linux, 
Windows or OS X.

#### Features

Among its major features:

* A NDDataset object embedding array of data with labeled axes and metadata.
* A Project manager to work on multiple NDDataset simultaneously.
* Physical Units for NDDataset.
* Mathematical operations over NDDataset such addition, multiplication and many more ...
* Import functions to read data from experiments or modeling programs ...
* Display functions such as plot for 1D or nD datasets ...
* Export functions to csv, xls formats ...
* Preprocessing functions such as baseline correction, automatic subtraction and many more ...
* Fitting capabilities for single or multiple datasets ...
* Exploratory analysis such as SVD, PCA, MCR_ALS, EFA ...

#### Documentation

The online Html documentation is available here:  [HTML documentation](https://www.spectrochempy.fr)

#### Issue Tracker

SpectroChemPy is still experimental and under active development. 
Its current design is subject to major changes, reorganizations, bugs and crashes!!!. 

You find a problem, want to suggest enhancements or want to look at the current issues and milestones, 
you can go there:  [Issue Tracker](https://github.com/spectrochempy/spectrochempy/issues)

#### Citing SpectroChemPy

When using SpectroChemPy for your own work, you are kindly requested to cite it this way:
```text
Arnaud Travert & Christian Fernandez, (2020) SpectroChemPy (Version 0.1). Zenodo. http://doi.org/10.5281/zenodo.3823841
```

#### Source repository 

The source are versioned using the git system and hosted on the GitHub platform: 
https://github.com/spectrochempy/spectrochempy

#### License

[CeCILL-B FREE SOFTWARE LICENSE AGREEMENT](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)


""",
                        ),
                    ]
            )
    )

def data_tab():

    upload = dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag & Drop or ',
                html.A('Select file', )
            ]),
            style={
                'height': '30px',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
            },
            # Allow multiple files to be uploaded
            multiple=True
    )

    upload_form = dbc.FormGroup(
            [
                dbc.Label("Input data"),
                upload,
            ],
    )

    return dbc.Card(
            dbc.CardBody(
                    [
                        upload_form,
                        html.Div(
                        dbc.Button(
                                "Clear",
                                color="danger",
                                id='input_clear_data',
                        ),
                        id='output-button-clear-data'),
                    ]
            ),
            className="mt-3",
    )

def layout():

    tab2_content = dbc.Card(
        dbc.CardBody(
            [
                html.P("This is tab 2!", className="card-text"),
                dbc.Button("Don't click here", color="success"),
            ]
        ),
        className="mt-3",
    )


    tabs = dbc.Tabs(
        [
            dbc.Tab(about_tab(), label="About"),
            dbc.Tab(data_tab(), label="Data"),
            dbc.Tab(tab2_content, label="Tab 2", disabled=True),
        ]
    )

    return  html.Div([
            html.P(""),
            dbc.Container(
            dbc.Row(  # 12 columns
                  [
                      dbc.Col(
                              tabs,
                              width=4,
                      ),
                      dbc.Col(
                              dcc.Loading(
                                      html.Div(
                                              id='output-data-upload',
                                      ),
                              ),
                              width=8,
                      ),
                  ],
            ),
            fluid=True,
            style={"font-size":'.75em'},
    ),
    ]
            ,)


# Callbacks ............................................................................................................

def callbacks(_app):

    def parse_upload_contents(contents, filename, date):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        ds = scp.NDDataset.read(filename, content=decoded)
        # get the matplotlib image
        fig = Figure()
        ds.plot(fig=fig)
        buf=io.BytesIO()
        ds.fig.savefig(buf, format='png')
        ds.close_figure()
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return  html.Img(src=f'data:image/png;base64,{data}')

    # add upload callback
    @_app.callback([Output('output-data-upload', 'children'),
                    Output('output-button-clear-data', 'style')],
                  [Input('upload-data', 'contents')],
                  [State('upload-data', 'filename'),
                   State('upload-data', 'last_modified')])
    def update_data_output(list_of_contents, list_of_names, list_of_dates):
        children=[]
        style={'visibility':'hidden'}
        if list_of_contents is not None:
            children = [
                parse_upload_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            style={'visibility':'visible'}
        return children, style

    # add callback for toggling the collapse on small screens
    @_app.callback(
            Output("navbar-collapse", "is_open"),
            [Input("navbar-toggler", "n_clicks")],
            [State("navbar-collapse", "is_open")],
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

app = run_standalone_app(layout, callbacks, __file__, menus=menus)
server = app.server

app.run_server(debug=True, port=8049)


