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

from dash_util import create_gif_card

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ------------------------------------------------------------------------------
# App layout

# callback options
row_list = [
    html.Hr(),
    dbc.Row([
        dbc.Col(html.H6("Select index of example: "), width=2),
        dbc.Col(dcc.Input(
            id="example_ind", type="number", placeholder="example index",
            min=1, max=200, step=1,
        ), width=0.5),
        dbc.Col(html.H6("Set number of nearest neighbors: "), width=2),
        dbc.Col(dcc.Input(
            id="num_NN", type="number", placeholder="number of nearest neighbors",
            min=1, max=10, step=1,
        ), width=0.5),
        dbc.Col(html.H6("Choose which model to visualize: "), width=2),
        dbc.Col(dcc.Dropdown(id='model_path',
            options=[
                {'label': 'log_train_earlyaction_hyper_v1_poincare_kinetics_lr4', 'value': '/proj/vondrick/didac/code/DPC/logs/log_train_earlyaction_hyper_v1_poincare_kinetics_lr4/20201023_151021/embeds/NN'},
                    ], value='/proj/vondrick/didac/code/DPC/logs/log_train_earlyaction_hyper_v1_poincare_kinetics_lr4/20201023_151021/embeds/NN'
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
    [Input("example_ind", "value"), Input("num_NN", "value"), dash.dependencies.Input('model_path', 'value')],
)
def number_render(example_ind, num_NN, model_path):
    
    # load gif_info dictionary
    with open(os.path.join(model_path, 'gif_info.json')) as f:
        gif_info = json.load(f)
        
    # original video
    gif_path = os.path.join(model_path, 'seq-%d.gif' % example_ind)
    action = gif_info['seq-%d' % example_ind]['action']
    row_list = [html.H5(["This is the visualization of No. %d examples" % (example_ind+1)]), html.Hr()]
    row_list.append(dbc.Row([dbc.Col([create_gif_card(gif_path, title='original video', action=action)], width=2)]))


    # predicted nearest neighbor of each time step
    for step in range(7): # num_pred
        row_list.append(html.H5(["No. %d clips seen during prediction" % (step+1)]))
        col_list = []
        for K in range(num_NN): # num_NN
            gif_path = os.path.join(model_path, 'seq-%d_seen-%d_pred-%d_K-%d.gif' % (example_ind, step+1, 1, K))
            action = gif_info['seq-%d_seen-%d_pred-%d_K-%d' % (example_ind, step+1, 1, K)]['action']
            col_list.append(dbc.Col([
                create_gif_card(gif_path, title='No. %d nearest neighbor' % K, action=action)
            ], width=2))
        row_list.append(dbc.Row(col_list))
    return row_list


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, port=10086)