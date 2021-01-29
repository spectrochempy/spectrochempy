# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
"""
This module implements the DAsh callbacks.
"""

__all__ = []

import base64
import json
from dash import no_update, callback_context
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import spectrochempy as scp


class Callbacks(object):

    @staticmethod
    def parse_upload_contents(filename, contents, single=False):
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
                dbc.Checkbox(id=f"dataset-checkbox-{i}", style={
                        'float': 'right'
                        }, checked=True),
                *self.dataset_itemlist(ds)]) for i, ds in enumerate(datasets)]

    def uploaded_dataset_list(self, *datasets):
        #
        # TODO: active can be set to True for highligthing selected File (however for the moment we limit to one
        #  component)
        # TODO: write  a callback for each entry
        #
        all_datasets = [
                dbc.ListGroup(self.dataset_list(*datasets))
                ]
        return all_datasets

    @staticmethod
    def set_ROI_and_offset(ds, xrf, yrf, xrl, xru, yrl, yru):

        dimx = ds.dims[-1]
        x = getattr(ds, dimx)

        x.offset = float(xrf) if xrf else 0
        ds.history = f">>> ds.{dimx}.offset = {x.offset}')"
        x.roi = [float(xrl) if xrl else 0, float(xru) if xru else 0]
        ds.history = f">>> ds.{dimx}.roi = {x.roi}"

        if len(ds.dims) > 1:

            dimy = ds.dims[-2]
            y = getattr(ds, dimy)

            y.offset = float(yrf) if yrf else 0
            ds.history = f">>> ds.{dimy}.offset = {y.offset}"
            y.roi = float(yrl) if yrl else 0, float(yru) if yru else 0
            ds.history = f">>> ds.{dimy}.roi = {y.roi}"

        return ds

    def add_callbacks(self, app):
        """ Add all application calbacks """

        # # ------------
        # # DATA UPLOAD
        # # ------------
        # @app.callback(Output('original-data', 'data'),
        #               Input('upload-data', 'contents'),
        #               State('upload-data', 'filename'))
        # def data_upload(list_of_contents, list_of_names):
        #     # Store data in components
        #
        #     data = {}  # the data must be JSON serializable
        #
        #     if list_of_contents is not None:
        #         # data uploaded
        #         z = zip(list_of_names, list_of_contents)
        #         data = {n: self.parse_upload_contents(n, c) for n, c in z}
        #     else:
        #         raise PreventUpdate
        #     return data

        # --------------
        # PROJECT UPLOAD
        # --------------

        @app.callback(Output('original-data', 'data'),
                      Input('upload-project', 'contents'),
                      State('upload-project', 'filename'))
        def project_upload(content, name):
            # Store project data in components

            data = {}  # the data must be JSON serializable

            if content is not None:
                # data uploaded
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                pj = scp.Project.load(content=decoded)
                data = pj.to_json()
            else:
                raise PreventUpdate
            return data

        # ----------------------
        # SAVE INTERMEDIATE DATA
        # ----------------------
        @app.callback([Output('intermediate-data', 'data'),
                       Output('actual-x-roi', 'children'),
                       Output('actual-y-roi', 'children'),
                       Output('action-history', 'data')],
                      [  # roi
                              Input('x-roi-lower', 'value'),
                              Input('x-roi-upper', 'value'),
                              Input('y-roi-lower', 'value'),
                              Input('y-roi-upper', 'value'),
                              Input('x-offset', 'value'),
                              Input('y-offset', 'value'),

                              Input('original-data', 'modified_timestamp'),  # new original data
                              Input('confirm-mask', 'submit_n_clicks'),  # modify masks
                              Input('graph', 'selectedData'),  # data for x-masks
                              Input('graph', 'clickData'),  # data for y-mask
                              Input('graph-selector', 'value'),  # transposed case?
                              ],
                      [State('original-data', 'data'),
                       State('intermediate-data', 'data'),
                       State('action-history', 'data')])
        def save_intermediate_data(xrl, xru, yrl, yru, xrf, yrf,
                                   ts, submit_mask, selectedData, clickData, selector,
                                   data, saved, history):
            """
            Main callback for updating the figure and some dash components
            """
            # no original data? probably not yet uplaoded... exit without updating
            if data is None:
                raise PreventUpdate

            project = scp.Project.from_json(data)
            # extract the project information from original data store
            # datasets = [scp.read_json(content=c.encode()) for n, c in data.items()]

            # show processed flag
            processed = True if 'Processed' in selector else False

            # if we want to show the original data:
            # -------------------------------------
            if not processed:
                ds = datasets[0]

                # set current ROI and offset if necessary
                self.set_ROI_and_offset(ds, xrf, yrf, xrl, xru, yrl, yru)

                # make output
                datasets[0] = ds
                data = {ds.filename: ds.write_json(to_string=True) for ds in datasets}
                return data, no_update, no_update, no_update

            # else we will output processed data
            # ----------------------------------
            if saved:
                datasets = [scp.read_json(content=c.encode()) for n, c in saved.items()]

            # for now we work with only one spectrum
            ds = datasets[0]
            dss = ds.copy()  # for further comparison of changes

            # set current ROI and offset if necessary
            self.set_ROI_and_offset(ds, xrf, yrf, xrl, xru, yrl, yru)

            # show transposed?
            transposed = True if 'Transposed' in selector else False

            # apply masks
            if submit_mask and selectedData:
                # set x-masks
                range = selectedData.get('range', None)
                if range is not None:
                    x_selection = range['x']
                    if not transposed:
                        ds[:, x_selection[1]:x_selection[0]] = scp.MASKED
                    else:
                        ds[x_selection[1]:x_selection[0]] = scp.MASKED

            # create output
            if ds == dss:
                # no change
                raise PreventUpdate

            datasets[0] = ds
            newdata = {ds.filename: ds.write_json(to_string=True) for ds in datasets}
            actx = f'Actual roi: {ds.x.roi_values[0]:~P} -> {ds.x.roi_values[1]:~P}'
            acty = f'Actual roi: {ds.y.roi_values[0]:~P} -> {ds.y.roi_values[1]:~P}'

            # update history
            # history is designed to be a list of python function,
            # able to be run later to reproduce all the transformations done
            ctx = callback_context.triggered
            if 'original-data' in ctx[0]['prop_id']:
                # Data just uploaded in Dash
                history = [f">>> ds = scp.read('{ds.filename}')"]
            else:
                # parameter changes
                for item in ctx:
                    par = item['prop_id']

            return newdata, actx, acty, history

        # -----------------------
        # UPDATE DATA TAB CONTENT
        # -----------------------
        @app.callback([Output('current-data', 'children'),
                       Output('show-project', 'style'),
                       Output('show-current-data', 'style'),
                       Output('show-graph', 'is_open'),
                       Output('x-roi', 'children'),
                       Output('y-roi', 'children'),
                       Output('x-roi-lower', 'value'),
                       Output('x-roi-upper', 'value'),
                       Output('y-roi-lower', 'value'),
                       Output('y-roi-upper', 'value'),
                       Output('x-roi-units', 'children'),
                       Output('y-roi-units', 'children'),
                       Output('x-offset', 'value'),
                       Output('y-offset', 'value')],
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
            style = dict({
                    'display': 'none'
                    })

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
                style = dict({
                        'display': 'block'
                        })

            return (dataloaded, style, style, not is_open,
                    xr, yr, xrl, xru, yrl, yru,
                    xr_units, yr_units, xro, yro)

        # -------------
        # UPDATE FIGURE
        # -------------

        @app.callback(Output('graph', 'figure'),
                      [Input('intermediate-data', 'data'),
                       Input('graph-selector', 'value'),
                       # change on the type of figure to display (processed, transposed...)
                       Input('graph-optimisation', 'value'),  # change the optimisation level
                       Input('zoom-reset', 'n_clicks'),  # reset Zoom from button
                       Input('cmap-select', 'value'),
                       Input('select-mask', 'n_clicks')

                       # Input('graph', 'relayoutData')
                       ])
        def update_figure(data,
                          selector,
                          optim,
                          zoomreset,
                          cmap,
                          dragmode,
                          ):  # relayout):

            if data is None:
                raise PreventUpdate

            datasets = [scp.read_json(content=c.encode()) for n, c in data.items()]

            ds = datasets[0]

            dragmode = 'zoom' if dragmode is None or dragmode % 2 == 0 else 'select'
            figure = ds.plot(use_plotly=True,
                             selector=selector,
                             optimisation=optim,
                             zoomreset=zoomreset,
                             cmap=cmap,
                             dragmode=dragmode,
                             )

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
        # DISPLAY CONTROL
        # -------------------
        @app.callback(
                [Output("close-data", "style"),
                 Output('data-tab', 'disabled'),
                 Output('graph-tab', 'disabled'),
                 Output('processing-tab', 'disabled')],
                [Input('current-data', 'children')])
        def tab_display_control(children):
            if not children:
                return dict({
                        'display': 'none'
                        }), True, True, True
            else:
                return dict({
                        'display': 'block'
                        }), False, False, False

        # ----------------------------------
        # MODIFY CLOSING/OPENING CARD BUTTON
        # ----------------------------------
        for item in ['roi', 'current-data', 'mask',
                     'layout', 'xaxis', 'zaxis',
                     'baseline', 'peakpicking', 'subtraction']:
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

        # ------------
        # MENU PROJECT
        # ------------
        @app.callback(
                [Output(f"project-name", "placeholder"),
                 Output(f"project-name", "disabled"),
                 Output(f"project-close", "disabled"),
                 Output(f"project-save", "disabled")],
                [Input(f"project-new", "n_clicks"),
                 Input(f"project-open", "n_clicks")]
                )
        def on_menu_project(new, open):
            # ctx = callback_context.triggered
            if new is None and open is None:
                raise PreventUpdate

            ctx = callback_context.triggered
            if ctx and 'new' in ctx[0]['prop_id'] and ctx[0]['value'] is not None:
                return 'Untitled (edit to chnge this name)', False, False, False
            elif ctx and 'new' in ctx[0]['prop_id'] and ctx[0]['value'] is not None:
                return

        # Set masks
        # @app.callback()

        # GRAPH SELECTION AND HOVER

        # @app.callback(
        #         Output('text-data', 'children'),
        #         [Input('graph', 'hoverData')])
        # def display_hover_data(hoverData):
        #     return json.dumps(hoverData, indent=2)
        #
        #
        # @app.callback(
        #         Output('text-data', 'children'),
        #         [Input('graph', 'clickData')])
        # def display_click_data(clickData):
        #     return json.dumps(clickData, indent=2)
        #
        #
        # @app.callback(
        #         Output('text-data', 'children'),
        #         [Input('graph', 'selectedData')])
        # def display_selected_data(selectedData):
        #     return json.dumps(selectedData, indent=2)

        @app.callback(
                [Output('text-data', 'children'),
                 Output('confirm-mask', 'displayed'),
                 Output('confirm-mask', 'submit_n_clicks'),
                 Output('confirm-mask', 'message')],
                [Input('graph', 'relayoutData'),
                 Input('graph', 'hoverData'),
                 Input('graph', 'selectedData'),
                 Input('graph', 'clickData')
                 ])
        def display_relayout_data(relayoutData, hoverData, selectedData, clickData):
            ctx = callback_context
            text = json.dumps(hoverData, indent=2) + json.dumps(relayoutData, indent=2) + json.dumps(selectedData,
                                                                                                     indent=2) + \
                   json.dumps(
                           clickData, indent=2)
            confirm = False
            message = ''
            nclicks = no_update
            if ctx.triggered[0]['prop_id'] == 'graph.selectedData':
                range = selectedData.get('range', None)
                if range is not None:
                    x_selection = range['x']
                    message = f'Are you sure you want to mask data in the {x_selection} region?'
                    confirm = True
                    nclicks = 0

            return text, confirm, nclicks, message

        #
        # Mask selection button aspect
        #

        @app.callback(
                [Output('select-mask', 'children'),
                 Output('select-mask', 'color'), ],
                [Input('select-mask', 'n_clicks'), ]
                )
        def mask_buttton_aspect(n):
            if n is None or n % 2 == 0:
                return "Select mask", "secondary"
            else:
                return "Stop mask selection", "danger"
