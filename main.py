from dash import Dash, html, dcc, Input, Output, State, ctx, ALL, exceptions
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import random
import requests
import io as io_buffer
from PIL import Image
import re
import dash
import pandas as pd  # Pour l'export Excel

# Configuration GitHub
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

def calculate_area(coords):
    if len(coords) < 3:
        return 0
    x, y = zip(*coords)
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(coords) - 1)))

# Figure de démarrage (affichée tant qu'aucun fichier n'est sélectionné)
scatter_fig = go.Figure(
    go.Scattergl(
        x=np.random.randn(1000),
        y=np.random.randn(1000),
        mode='markers',
        marker=dict(
            color=random.sample(['#ecf0f1'] * 500 + ["#3498db"] * 500, 1000),
            line_width=1
        )
    )
)
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
classification_options = ["tache", "plaque", "atrophique",  "incertaine"]

app.layout = dbc.Container([
    html.Div([
        html.Div([
            html.H1([
                html.Span("Instructions d'utilisation")
            ]),
            html.P("1. Sélectionnez un fichier à analyser dans le menu déroulant."),
            html.P("2. Utilisez l'outil de dessin pour détourer une tache sur l'image."),
            html.P("3. Vous pouvez réinitialiser le détourage'."),
            html.P("4. Attribuez une classification à chaque zone en cliquant sur l'un des boutons proposés après le détourage."),
            html.P("5. Vous pouvez modifier une classification en selectionnant une Zone existante à partir du menu déroulant"),
            html.P("6. Vous pouvez modifier la taille d'une tache en la sélectionannt sur le dessin"),
            html.P( "7. Cliquez sur 'Exporter vers Excel' pour télécharger un tableur contenant un résumé des zones pour une image")
        ],style={'margin-top': '20px', 'margin-bottom': '20px', 'padding': '10px'}),
        # Zone d'instructions pour guider l'utilisateur
        html.Div([
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': f, 'value': f} for f in filenames],
                placeholder='Sélectionnez un fichier à analyser'
            ),
            html.Br(),
            html.P("Classification :"),
            dbc.ButtonGroup([
                dbc.Button(opt, id={"type": "classify-button", "index": opt}, color="secondary")
                for opt in classification_options
            ], vertical=False, className="mb-2"),
            # Dropdown pour sélectionner une zone existante
            dcc.Dropdown(
                id="zone-selector",
                options=[],
                placeholder="Sélectionnez une zone à reclassifier",
                style={"margin-top": "10px"}
            ),
            html.Br(),
            dbc.Button("Réinitialiser les annotations", id="reset-button", color="danger", className="mb-2"),
            html.Br(),
            # Bouton Exportation vers Excel
            dbc.Button("Exporter vers Excel", id="export-button", color="primary", className="mb-2"),
            # Composant Download pour le téléchargement du fichier
            dcc.Download(id="download-dataframe-xlsx"),
            # Champ caché pour les raccourcis clavier (ici non utilisé, mais conservé)
            dcc.Input(id="key-capture", type="text",
                      style={"opacity": 0, "position": "absolute"},
                      autoFocus=True),
            dcc.Store(id="stored-shapes", data=[]),
            html.Div(id='output-text')
        ], style={'margin-left': 5, 'margin-right': 5}),
    ], style={'width': 340, 'margin-left': 35, 'margin-top': 35, 'margin-bottom': 35}),
    html.Div([
        dcc.Graph(
            id='fig-image',
            # Configuration pour conserver les outils de dessin/effacement
            config={"modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"], "displaylogo": False},
            style={'width': 990, 'height': 730}
        ),
        html.Div(id='output-area', style={'color': 'white', 'margin-top': '10px'})
    ], style={'width': 990, 'margin-top': 35, 'margin-right': 35, 'margin-bottom': 35, 'display': 'flex'})
],
    fluid=True,
    style={'display': 'flex'},
    className='dashboard-container'
)

def generate_figure(image):
    fig = px.imshow(image)
    # Configuration du mode de dessin
    fig.update_layout(
        dragmode="drawclosedpath",
        paper_bgcolor='black',
        plot_bgcolor='black',
        width=image.width,
        height=image.height,
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=[]
    )
    return fig

# Callback pour générer ou mettre à jour le graphique (figure)
@app.callback(
    Output("fig-image", "figure"),
    Input("file-dropdown", "value"),
    Input("reset-button", "n_clicks"),
    Input("stored-shapes", "data"),
    State("fig-image", "figure")
)
def update_figure(file_val, reset_clicks, stored_shapes, current_fig):
    trigger = ctx.triggered_id
    if trigger in ["file-dropdown", "reset-button"]:
        if not file_val:
            fig = scatter_fig
        else:
            try:
                image_url = get_image_url(file_val)
                response = requests.get(image_url)
                image = Image.open(io_buffer.BytesIO(response.content))
                image = image.resize((790, 790))
                fig = generate_figure(image)
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {str(e)}")
    else:
        fig = current_fig if current_fig is not None else scatter_fig

    # Mise à jour des annotations (numéros de zone)
    if stored_shapes is not None:
        def centroid(coords):
            if not coords:
                return 0, 0
            avg_x = sum(x for x, y in coords) / len(coords)
            avg_y = sum(y for x, y in coords) / len(coords)
            return avg_x, avg_y

        annotations = []
        for i, shape in enumerate(stored_shapes):
            path_str = shape.get("path", "")
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
            coords = []
            try:
                coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
            except Exception:
                continue
            cx, cy = centroid(coords)
            annotations.append(dict(
                x=cx,
                y=cy,
                text=str(i + 1),
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
                font=dict(color="white", size=12)
            ))
        fig["layout"]["annotations"] = annotations
    return fig

# Callback pour mettre à jour les données stockées et l'affichage des aires
@app.callback(
    Output("stored-shapes", "data"),
    Output("output-area", "children"),
    Output("key-capture", "value"),
    Input("fig-image", "relayoutData"),
    Input("reset-button", "n_clicks"),
    Input({"type": "classify-button", "index": ALL}, "n_clicks"),
    Input("key-capture", "value"),
    State("stored-shapes", "data"),
    State("zone-selector", "value"),
    prevent_initial_call=True
)
def update_shapes(relayout_data, reset_clicks, classify_clicks, key_value, stored_shapes, selected_zone):
    if stored_shapes is None:
        stored_shapes = []
    trigger = ctx.triggered_id

    if trigger == "reset-button":
        return [], "Annotations réinitialisées.", ""

    if isinstance(trigger, dict) and trigger.get("type") == "classify-button":
        label = trigger["index"]
        if selected_zone is not None and selected_zone < len(stored_shapes):
            stored_shapes[selected_zone]["customdata"] = label
        elif stored_shapes:
            stored_shapes[-1]["customdata"] = label

    if trigger == "key-capture":
        key_value = ""

    if relayout_data:
        if "shapes" in relayout_data:
            new_shapes = relayout_data["shapes"]
            if len(new_shapes) == len(stored_shapes):
                for i, new_shape in enumerate(new_shapes):
                    new_shape["customdata"] = stored_shapes[i].get("customdata", "non classé")
            else:
                for i, new_shape in enumerate(new_shapes):
                    if i < len(stored_shapes):
                        new_shape["customdata"] = stored_shapes[i].get("customdata", "non classé")
                    else:
                        new_shape["customdata"] = "non classé"
            stored_shapes = new_shapes

    areas = []
    for i, shape in enumerate(stored_shapes):
        path_str = shape.get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
            area = calculate_area(coords)
            lab = shape.get("customdata", "non classé")
            areas.append(f"Zone {i + 1} : {area:.2f} pixels² ({lab})")
        except Exception as e:
            areas.append(f"Zone {i + 1} : erreur ({e})")

    return stored_shapes, html.Ul([html.Li(a) for a in areas]), key_value

# Callback pour mettre à jour les options du dropdown "zone-selector"
@app.callback(
    Output("zone-selector", "options"),
    Input("stored-shapes", "data")
)
def update_zone_selector_options(stored_shapes):
    if stored_shapes is None:
        return []
    return [{"label": f"Zone {i + 1}", "value": i} for i in range(len(stored_shapes))]

# Callback pour exporter les données vers Excel
@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("export-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    prevent_initial_call=True
)
def export_to_excel(n_clicks, stored_shapes, file_val):
    if not n_clicks or not stored_shapes:
        return dash.no_update

    # Préparer les données pour chaque zone
    rows = []

    def calc_centroid(path_str):
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
            if coords:
                avg_x = sum(x for x, y in coords) / len(coords)
                avg_y = sum(y for x, y in coords) / len(coords)
                return avg_x, avg_y
            else:
                return None, None
        except Exception:
            return None, None

    for i, shape in enumerate(stored_shapes):
        path_str = shape.get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
            area = calculate_area(coords)
        except Exception:
            area = None
        cx, cy = calc_centroid(path_str)
        classification = shape.get("customdata", "non classé")
        rows.append({
            "Zone": i + 1,
            "Aire (pixels²)": area,
            "Centroid X": cx,
            "Centroid Y": cy,
            "Classification": classification
        })

    df = pd.DataFrame(rows)
    filename = f"{file_val.split('.')[0]}.xlsx" if file_val else "export.xlsx"

    def to_excel(bytes_io):
        with pd.ExcelWriter(bytes_io, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Zones")
    return dcc.send_bytes(to_excel, filename)

if __name__ == '__main__':
    app.run(debug=False)
