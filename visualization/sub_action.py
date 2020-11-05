import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import json
import sys

from dash_util import create_gif_card

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ------------------------------------------------------------------------------
# App layout

# callback options
row_list = [
    html.Hr(),
    dbc.Row([
        dbc.Col(html.H6("Select pages: "), width=2),
        dbc.Col(dcc.Input(
            id="page_ind", type="number", placeholder="index of page",
            min=1, max=100, step=1, value=1
        ), width=0.5),
        dbc.Col(html.H6("Set number of nearest neighbors: "), width=2),
        dbc.Col(dcc.Input(
            id="num_NN", type="number", placeholder="number of nearest neighbors",
            min=1, max=10, step=1, value=3
        ), width=0.5),
        dbc.Col(html.H6("Set number of examples per page: "), width=2),
        dbc.Col(dcc.Input(
            id="num_vid", type="number", placeholder="number of videos per page",
            min=1, max=30, step=1, value=10
        ), width=0.5),
        dbc.Col(html.H6("Set number of steps to predict: "), width=2),
        dbc.Col(dcc.Input(
            id="pred_step", type="number", placeholder="number of predicted steps",
            min=1, max=3, step=1, value=1
        ), width=0.5),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col(html.H6("Choose which model to visualize: "), width=2),
        dbc.Col(dcc.Dropdown(id='model_path',
            options=[
                {'label': 'log_train_dpc_hyper_v1_poincare_kinetics', 'value': '/proj/vondrick/didac/code/DPC/logs/log_train_dpc_hyper_v1_poincare_kinetics/20201019_195227/embeds/NN'},
                    ], value='/proj/vondrick/didac/code/DPC/logs/log_train_dpc_hyper_v1_poincare_kinetics/20201019_195227/embeds/NN'
                    ),
               ),
    ]),
    html.Hr(),
    html.Div(id='output_list')
]

app.layout = html.Div(row_list)

            
'''
list of callback elements needed:
1. number of nearest neighbors (default=5)
2. index of examples
3. model_path
'''
            
# ------------------------------------------------------------------------------
@app.callback(
    Output("output_list", "children"),
    [Input("page_ind", "value"), Input("num_NN", "value"), Input("num_vid", "value"),\
     Input("pred_step", "value"), dash.dependencies.Input('model_path', 'value')],
)
def number_render(page_ind, num_NN, num_vid, pred_step, model_path):
    
    # load gif_info dictionary
    with open(os.path.join(model_path, 'gif_info.json')) as f:
        gif_info = json.load(f)
    
    row_list = []
    for vid_ind in range(num_vid):
        # original video
        col_list = []
        example_ind = page_ind * num_vid + vid_ind
        gif_path = os.path.join(model_path, 'seq-%d.gif' % example_ind)
        action = gif_info['seq-%d' % example_ind]['action']
        col_list.append(dbc.Col([create_gif_card(gif_path, title='original video', action=action)], width=2))
        
        # predicted nearest neighbor of each time step
        for K in range(num_NN): # num_NN
            gif_path = os.path.join(model_path, 'seq-%d_seen-%d_pred-%d_K-%d.gif' % (example_ind, 5, pred_step, K))
            action = gif_info['seq-%d_seen-%d_pred-%d_K-%d' % (example_ind, 5, pred_step, K)]['action']
            col_list.append(dbc.Col([
                create_gif_card(gif_path, title='No. %d nearest neighbor' % K, action=action)
            ], width=2))
        row_list.append(dbc.Row(col_list))
    return row_list


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, port=sys.argv[1], host='0.0.0.0')