# adapted from dash_bio

import base64
import os

#from jupyter_dash import JupyterDash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc
# see https://dash-bootstrap-components.opensource.faculty.ai/docs




def logo(filename):

    SCPY_LOGO = os.path.join(os.path.dirname(os.path.abspath(filename)), 'assets','scpy_logo.png')
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




def  navbar(filename, **kwargs):

    _menus = kwargs.get('menus',None)
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
                dbc.NavbarToggler(
                    id="navbar-toggler",
                ),
                nav,
            ],
            color="dark",
            dark=True,
            sticky='top',
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

    theme = kwargs.pop('theme', dbc.themes.JOURNAL)
    #CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN, LUX, MATERIA, MINTY, PULSE, SANDSTONE, SIMPLEX, SKETCHY, SLATE, SOLAR, SPACELAB, SUPERHERO, UNITED, YETI

    app = Dash(__name__, external_stylesheets=[theme])
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True

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
