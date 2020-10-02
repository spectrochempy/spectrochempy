# -*- coding: utf-8 -*
# %% [markdown]
# # SpectroChemPy GUI: Dash Application

# %% [markdown]
# Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                        

# %% [markdown]
# **Note:** Run this app with the command `scpy` in a terminal and then visit http://127.0.0.1:8050/ in your web
# browser.

# %%
import base64
import io
import os

# from jupyter_dash import JupyterDash
from dash import Dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from matplotlib.figure import Figure

import spectrochempy as scp

import webbrowser
from threading import Timer

import plotly.graph_objects as go
from matplotlib import cm
import numpy as np

# see https://dash-bootstrap-components.opensource.faculty.ai/docs
import dash_defer_js_import as dji

def description():
	return 'SpectroChemPy DASH application.'


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
			[dcc.Store(id='session-data', storage_type='session'),
			 logo(filename),
			 navbarbrand(),
			 dbc.NavbarToggler(
					 id="navbar-toggler",
			 ),
			 nav,
			 ],
			color="dark",
			dark=True,
			#sticky='top',
	)


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
	print(__name__)
	app = Dash(__name__, title='Dash-SpectroChemPy',
			   external_stylesheets=[theme])
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
						children=page_layout
				)
			],
	)

# .............................................................................
def menus():
	dropdown = dbc.DropdownMenu(
			children=[dbc.DropdownMenuItem("Open", id='menu_open'), dbc.DropdownMenuItem("Close", id='menu_close'),
					  dbc.DropdownMenuItem("Save", id='menu_save'), dbc.DropdownMenuItem(divider=True),
					  dbc.DropdownMenuItem("Preferences", id='menu_preferences'), ], nav=True, in_navbar=True,
			label="File", )
	return [dropdown]


# .............................................................................
def about_tab():
	return dbc.Card(dbc.CardBody([

		dcc.Markdown("""
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


""", ), ]))


def data_tab():
	upload = dcc.Upload(id='upload-data',
						children=html.Div(['Drag & Drop or ', html.A('Select file')]),
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
						multiple=True)

	upload_form = dbc.Card(dbc.CardBody(
				[
					html.H6("Input data"),
			 		upload
				]
			))

	return dbc.Card(dbc.CardBody(
					[
						dbc.Collapse(upload_form, id='show-upload-form'),
						dbc.Collapse(html.Div(id='current-data'), id='show-current-data'),
						dcc.Loading(
								dbc.Collapse(
										dbc.Button("Close", color="danger",
												   style={'float': 'right', 'margin-top':'10px'},
												   id='close-data'),
										id='show-close-data'),
								type='circle'), ]
			))


def processing_tab():
	return dbc.Card(dbc.CardBody(
			[html.P("This is tab 2!", className="card-text"),
			 dbc.Button("Don't click here", color="success"), ]),
			className="mt-3", )


def layout():
	tabs = dbc.Tabs(
			[dbc.Tab(about_tab(), label="About", tab_id='tab-1'),
			 dbc.Tab(data_tab(), label="Data", tab_id='tab-2'),
			 dbc.Tab(processing_tab(), label="Processing", tab_id='tab-3', id='output-processing-tab')
			 # disabled=True),
			 ], style={'background-color': 'beige'}, active_tab='tab-2', )


	mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")
	refresh_plots = dji.Import(src="https://www.spectrochempy.fr/gui/redraw.js")

	return html.Div([
		html.P(""),
		dbc.Container(
				dbc.Row( # 12 columns
						[dbc.Col(tabs, width=4, ),
						 dbc.Col(
								 dcc.Loading(
										 dcc.Graph( id='graph'),
										 )
								 ),
						 ],
						),
			fluid=True, style={"font-size": '.75em'}, ),
		refresh_plots,
		mathjax_script,
	],)


def plotly_2D(*datasets):
	"""
	Create a plotly figure for each dataset

	Parameters
	----------
	datasets

	Returns
	-------

	"""
	figs = []
	for ds in datasets:
		x = ds.x.data
		ds.y -= ds.y[0]

		rgbfactor = 255/ds.y.values.max()
		cmap = np.array(cm.get_cmap('viridis').colors)*255

		data = []
		for i, row in enumerate(ds):
			trace =row.data.squeeze()
			name = str(row.y.values)
			k = int(row.y.values * rgbfactor)
			data.append(dict(x=x,
							 xaxis='x',
							 y=trace,
							 yaxis='y',
							 name=name,
							 mode='lines',
							 type='scatter',
							 line=dict(
									 color=f'rgb{tuple(cmap[k].astype("uint8"))}',
									 dash='solid',
									 width= 0.7),
							 )
						)

		layout = dict({'autosize': True,
					   'height':600,
					   'hovermode': 'closest',
					   'showlegend': False,
					   'xaxis': {'anchor': 'y',
								 #'domain': [0.0, 1.0],
								 'nticks': 7,
								 'range': [5999.556, 649.904],
								 'showgrid': False,
								 'side': 'bottom',
								 'tickfont': {'family':'Times', 'size': 15},
								 'ticks': 'inside',
								 'title': {'font': {'family':'Times',
													'color': '#000000'},
										   'text': r'$\text{Wavenumbers}\ /\ \text{cm}^\text{-1}$'},
								 'type': 'linear',
								 'zeroline': False},
					   'yaxis': {'anchor': 'x',
								 'domain': [0.0, 1.0],
								 'nticks': 9,
								 'range': [-0.14341419234871866, 6.1204590988159175],
								 'showgrid': False,
								 'side': 'left',
								 'tickfont': {'family':'Times','size': 15.0},
								 'ticks': 'inside',
								 'title': {'font': {'family':'Times','color': '#000000'},
										   'text': r'$\text{Absorbance}\ /\ \text{a.u.}$'},
								 'type': 'linear',
								 'zeroline': False}})


		# Create traces
		fig = go.Figure( data=data,
					     layout=go.Layout(layout)
					)
		figs.append(fig)

	return figs

# Callbacks ............................................................................................................

def callbacks(app):
	def parse_upload_contents(contents, filename):
		content_type, content_string = contents.split(',')
		decoded = base64.b64decode(content_string)
		ds = scp.NDDataset.read(filename, content=decoded)
		return ds

	def make_thumbnail(*datasets):
		# get the matplotlib image
		thumbnails = []
		for ds in datasets:
			fig = Figure()
			ds.plot(fig=fig)
			buf = io.BytesIO()
			ds.fig.savefig(buf, format='png')
			ds.close_figure()
			data = base64.b64encode(buf.getbuffer()).decode("ascii")

			thumbnails.append(dbc.Card(dbc.CardBody([
				html.H6(ds.filename, className="card-title"),
				html.Img(src=f'data:image/png;base64,{data}', style={'width': '75px'}),
				html.P(str(ds))
				]
			)))
		return thumbnails

	# add upload callback
	@app.callback(Output('session-data', 'data'),
				  [Input('upload-data', 'contents')],
				  [State('upload-data', 'filename')])
	def update_data_output(list_of_contents, list_of_names):
		# Store data in dcc.Store components

		data = {}  # must be JSON serializable

		if list_of_contents is not None:
			z = zip(list_of_names, list_of_contents)
			data = {n: c for n, c in z}

		else:
			# prevent the None callbacks is important with the store component.
			# you don't want to update the store for nothing.
			raise PreventUpdate
		return data

	# # output the data information with  thumbnail img.
	@app.callback([Output('current-data', 'children'),
				   Output('graph', 'figure'),
				   Output('show-upload-form', 'is_open'),
				   Output('show-current-data', 'is_open'),
				   Output('show-close-data', 'is_open')],
				  # Since we use the data prop in an output,
				  # we cannot get the initial data on load with the data prop.
				  # To counter this, you can use the modified_timestamp
				  # as Input and the data as State.
				  # This limitation is due to the initial None callbacks
				  # https://github.com/plotly/dash-renderer/pull/81
				  [Input('session-data', 'modified_timestamp')],
				  [State('session-data', 'data')])
	def on_data(ts, data):
		if ts is None:
			raise PreventUpdate

		thumbnails = None
		figure= go.Figure(data=None)
		is_open = True

		if data is not None:
			datasets = [parse_upload_contents(c, n) for n, c in data.items()]
			thumbnails = make_thumbnail(*datasets)
			figure = plotly_2D(*datasets)[0]
			is_open = False
		data_is_open = not is_open
		button_is_open = not is_open
		return thumbnails, figure, is_open, data_is_open, button_is_open

	# close data callback
	@app.callback(
			Output("session-data", "clear_data"),
			[Input("close-data", "n_clicks")]
			)
	def on_button_click(n):
		if n is None:
			return False
		else:
			return True

	# add callback for toggling the collapse on small screens
	# note that navbar is defined in dash_gui.py
	@app.callback(Output("navbar-collapse", "is_open"),
				  [Input("navbar-toggler", "n_clicks")],
				  [State("navbar-collapse", "is_open")], )
	def toggle_navbar_collapse(n, is_open):
		if n:
			return not is_open
		return is_open


# .............................................................................

def open_browser():
	webbrowser.open_new('http://127.0.0.1:8049/')

def main():
	app = run_standalone_app(layout, callbacks, __file__, menus=menus)
	server = app.server

	Timer(1, open_browser).start()
	app.run_server(debug=True, port=8049)

if __name__ == '__main__':
    main()