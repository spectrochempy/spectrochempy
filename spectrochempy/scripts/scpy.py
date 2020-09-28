# -*- coding: utf-8 -*


# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""This module is the main entry-point for the GUI application launched from the
terminal command line

"""
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import datetime
import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

import spectrochempy as scp
from spectrochempy.core import app, info_

def gui():

    # Initialisation ...................................................................................................
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                            'https://www.spectrochempy.fr/gui/scpy.css']
    dashapp = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    scp.set_loglevel("INFO")

    # Layout ...........................................................................................................
    dashapp.layout = html.Div(children=[
        html.Img(src="https://www.spectrochempy.fr/stable/_static/scpy.png", alt="Logo", className="logo"),
        html.H1(children='SpectroChemPy'),
        html.H3(children='Processing, analyzing and modeling spectroscopic data with Python, '),
        html.H3(children='© 2014 - 2020 - A.Travert & C.Fernandez @LCS [ contact@spectrochempy.fr ]'),
        html.Hr(),
        html.Div(children='''
            notes .... (to be completed)
        '''),
        html.Hr(),
        html.H2('Data to be analyzed:'),
        dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '30px',
                    'lineHeight': '30px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),
        html.Table(id='output-data-upload',
        ),
        dcc.Graph(
                id='example-graph',
                #figure=fig
        )
    ])

    def parse_contents(contents, filename, date):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        ds = scp.NDDataset.read(filename, content=decoded)
        # get the matplotlib image
        ds.plot()
        buf=io.BytesIO()
        ds.fig.savefig(buf, format='png')
        ds.close_figure()
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return html.Tr(
               [html.Td([html.Img(src=f'data:image/png;base64,{data}', className='thumbnail')]),
               html.Td([html.P(f'Filename : {filename}'), html.P(f'{str(ds)}')])]
        )

    @dashapp.callback(Output('output-data-upload', 'children'),
                  [Input('upload-data', 'contents')],
                  [State('upload-data', 'filename'),
                   State('upload-data', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children

    dashapp.run_server(debug=True)

# ======================================================================================================================
if __name__ == '__main__':
    gui( )
