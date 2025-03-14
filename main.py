from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import random
import requests
import io as io_buffer
from PIL import Image
import plotly.express as px
import re  # Import de regex pour l'extraction des coordonnées

# Configuration GitHub (dépôt public)
REPO_OWNER = "thomasb6"
REPO_NAME = "alpha-detouring-BirdCHIN"
FOLDER_PATH = "optos_jpg"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FOLDER_PATH}"

GITHUB_TOKEN = "ghp_nwTO1ndYrsxh9HxEJKi2QiZNDWGCSX?3?z?U?g?NP"
GITHUB_TOKEN = GITHUB_TOKEN.replace("?", "")


def get_filenames():
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(GITHUB_API_URL, headers=headers)
    if response.status_code == 200:
        return [file["name"] for file in response.json() if file["type"] == "file"]
    return []


def get_image_url(filename):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{FOLDER_PATH}/{filename}"


# Fonction pour calculer l'aire avec la formule de Shoelace
def calculate_area(coords):
    if len(coords) < 3:
        return 0  # Un polygone doit avoir au moins 3 points
    x, y = zip(*coords)
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(coords) - 1)))


# Scatter Plot initial (inchangé)
scatter_fig = go.Figure(
    go.Scattergl(x=np.random.randn(1000),
                 y=np.random.randn(1000),
                 mode='markers',
                 marker=dict(
                     color=random.sample(['#ecf0f1'] * 500 +
                                         ["#3498db"] * 500, 1000),
                     line_width=1)
                 ))

scatter_fig.update_layout(
    plot_bgcolor='#010103',
    width=790,
    height=790,
    xaxis_visible=False,
    yaxis_visible=False,
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0)
)

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

filenames = get_filenames()

app.layout = dbc.Container([
    html.Div([html.Div([
        html.H1([
            html.Span("Annotation"),
            html.Br(),
            html.Span("Taches Birdshot")
        ]),
        html.P("VIsion tranformers for Birdshot Evaluation")
    ],
        style={
            "vertical-alignment": "top",
            "height": 260
        }),
        html.Div([
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': f, 'value': f} for f in filenames],
                placeholder='Sélectionnez un fichier à analyser'
            ), html.Div(id='output-text')],
            style={
                'margin-left': 5,
                'margin-right': 5,
            }),
    ],
        style={
            'width': 340,
            'margin-left': 35,
            'margin-top': 35,
            'margin-bottom': 35
        }),
    html.Div(
        [html.Div(id='graph-container', style={'width': 990, 'height': 730}),
         html.Div(id='output-area', style={'color': 'white', 'margin-top': '10px'})],
        style={
            'width': 990,
            'margin-top': 35,
            'margin-right': 35,
            'margin-bottom': 35,
            'display': 'flex'
        })
],
    fluid=True,
    style={'display': 'flex'},
    className='dashboard-container')


@app.callback(
    Output('graph-container', 'children'),
    [Input('file-dropdown', 'value')]
)
def display_selected_file(selected_filename):
    if not selected_filename:
        return dcc.Graph(id="fig-image", figure=scatter_fig, style={'width': 790})

    try:
        image_url = get_image_url(selected_filename)
        response = requests.get(image_url)
        image = Image.open(io_buffer.BytesIO(response.content))
        image = image.resize((790, 790))

        fig = px.imshow(image)
        fig.update_layout(
            dragmode="drawclosedpath",
            paper_bgcolor='black',
            plot_bgcolor='black',
            width=image.width,  # Utilisation de la taille originale
            height=image.height,
            xaxis_visible=False,
            yaxis_visible=False,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        config = {"modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"]}
        return dcc.Graph(id="fig-image", figure=fig, config=config)
    except Exception as e:
        return html.Div(f'Error loading file: {str(e)}')


@app.callback(
    Output("output-area", "children"),
    Input("fig-image", "relayoutData")
)
def update_area(relayout_data):
    if relayout_data and "shapes" in relayout_data:
        shapes = relayout_data["shapes"]
        if shapes and "path" in shapes[-1]:
            path_str = shapes[-1]["path"]

            # Extraction des coordonnées avec regex
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)

            # Conversion en paires (x, y) et calcul de l'aire
            try:
                coords = [(float(matches[i]), float(matches[i + 1])) for i in range(0, len(matches), 2)]
                area = calculate_area(coords)
                return f"Aire: {area:.2f} pixels²"
            except (ValueError, IndexError) as e:
                return f"Erreur dans le calcul : {str(e)}"
    return "Aire: 0 pixels²"


if __name__ == '__main__':
    app.run_server(debug=False)
