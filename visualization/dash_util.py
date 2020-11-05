import base64
import dash_bootstrap_components as dbc

def create_gif_card(gif_path, title, action):
    gif_base64 = 'data:image/png;base64,{}'.format(base64.b64encode(open(gif_path, 'rb').read()).decode('ascii'))
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.CardHeader(title),
                dbc.CardImg(src=gif_base64, top=True),
                dbc.CardHeader(action),
            ]
        )
    )