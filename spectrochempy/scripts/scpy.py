# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
SpectroChemPy DashBoard: Dash Application

Note
----

    Run this app with the command `scpy` in a terminal and then visit http://127.0.0.1:8050/ in your web browser.

"""

import webbrowser
from threading import Timer

import dash
import dash_bootstrap_components as dbc

import spectrochempy as scp
from spectrochempy.dashboard import DashBoard

# .............................................................................
def make_standalone_app(dashboard, **kwargs):
    """
    Run a dash application as standalone

    Parameters
    ----------
    filename: name of the calling script
    kwargs: other optional parameters

    Returns
    -------
    app: Dash application

    """

    theme = kwargs.pop('theme', dbc.themes.BOOTSTRAP)

    app = dash.Dash(__name__, title='Dash-SpectroChemPy', external_stylesheets=[theme])
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
    app.layout = dashboard.app_page_layout()

    # Register all callbacks
    dashboard.callbacks(app)

    # return app object
    return app

def main():

    # create the standalone application
    dashboard = DashBoard(calling_script_name=__file__)
    app = make_standalone_app(dashboard)

    # Run flask server in dev mode
    port = scp.app.port
    debug = scp.get_loglevel() == scp.DEBUG
    for i in range(10):
        try:
            _open_browser = lambda: webbrowser.open_new(f'http://127.0.0.1:{port+i}/')
            Timer(1, _open_browser).start()
            app.run_server(debug=debug, port=port+i)
            break

        except OSError as e:
            if "Address already in use" in e.strerror:
                continue


if __name__ == '__main__':
    main()
