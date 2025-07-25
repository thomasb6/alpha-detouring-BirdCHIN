# ====================================================
# IMPORTS ET CONFIGURATIONS INITIALS
# ====================================================
from scipy.stats import norm
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import dash
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import random
import requests
import io as io_buffer
from PIL import Image
import re
import pandas as pd
import json
from dash.dependencies import ALL
import base64
from skimage import measure
from skimage import filters, color, feature
from sklearn.ensemble import RandomForestClassifier

# ====================================================
# CONFIGURATION DE L'ACCÈS AU RÉPERTOIRE GITHUB
# ====================================================
REPO_OWNER = "thomasb6"
REPO_NAME = "alpha-detouring-BirdCHIN"
FOLDER_PATH = "exemples"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FOLDER_PATH}"
GITHUB_TOKEN = "ghp_nwTO1ndY???????????rsxh9HxEJKi2QiZNDWGCSX?3?z?U?g?NP"
GITHUB_TOKEN = GITHUB_TOKEN.replace("?", "")

# ====================================================
# FONCTIONS AUXILIAIRES POUR LA GESTION DES IMAGES ET DES COORDONNÉES
# ====================================================

def load_image_any(img_id):
    """img_id = soit nom de fichier GitHub, soit string base64 data:image/..."""
    if isinstance(img_id, str) and img_id.startswith("data:image"):
        # Local upload
        content_type, content_string = img_id.split(',')
        return Image.open(io_buffer.BytesIO(base64.b64decode(content_string)))
    elif img_id:
        # GitHub filename
        url = get_image_url(img_id)
        return Image.open(io_buffer.BytesIO(requests.get(url).content))
    return None


def mask_to_shapes(mask, label_value=1, min_area=200, contour_tolerance=3.0):
    mask_bin = (np.array(mask) == label_value).astype(np.uint8)
    labeled = measure.label(mask_bin)
    shapes = []
    idx = 1
    for region in measure.regionprops(labeled):
        if region.area < min_area:
            continue
        contours = measure.find_contours(labeled == region.label, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=lambda c: len(c))
        # Simplification ici
        contour = measure.approximate_polygon(contour, tolerance=contour_tolerance)
        path = "M " + " L ".join(f"{float(y)},{float(x)}" for x, y in contour) + " Z"
        shapes.append({
            "type": "path",
            "path": path,
            "line": {"color": "yellow", "width": 2, "dash": "solid"},
            "customdata": "segmentation-auto",
            "customid": idx,
            "editable": True,
            "layer": "above"
        })
        idx += 1
    return shapes



def transform_coords(coords, zoom, rotation_deg, center):
    from math import cos, sin, radians
    rot = radians(rotation_deg)
    cx, cy = center
    out = []
    for x, y in coords:
        tx, ty = x - cx, y - cy
        tx, ty = tx * zoom, ty * zoom
        rx = tx * cos(rot) - ty * sin(rot)
        ry = tx * sin(rot) + ty * cos(rot)
        out.append((rx + cx, ry + cy))
    return out

def transform_shape(shape, zoom, rotation_deg, center):
    s = shape.copy()
    if s.get("type") == "circle":
        coords = circle_to_coords(s)
        coords_t = transform_coords(coords, zoom, rotation_deg, center)
        xs = [pt[0] for pt in coords_t]
        ys = [pt[1] for pt in coords_t]
        s["x0"], s["x1"] = min(xs), max(xs)
        s["y0"], s["y1"] = min(ys), max(ys)
    elif "path" in s:
        import re
        path_str = s["path"]
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
        coords_t = transform_coords(coords, zoom, rotation_deg, center)
        if coords_t:
            path = "M " + " L ".join(f"{x},{y}" for x, y in coords_t) + " Z"
            s["path"] = path
    return s

def get_filenames(path=FOLDER_PATH):
    """
    Récupère récursivement la liste des fichiers images dans un dossier GitHub, y compris les sous-dossiers.

    Args:
        path (str): Le chemin relatif dans le dépôt GitHub.

    Returns:
        list: Liste des chemins relatifs vers les fichiers images.
    """
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []

    items = response.json()
    filenames = []
    for item in items:
        if item["type"] == "file":
            filenames.append(item["path"])  # on garde le chemin relatif
        elif item["type"] == "dir":
            filenames.extend(get_filenames(item["path"]))  # récursif
    return filenames


def get_image_url(filepath):
    """
    Construit l'URL brute vers une image GitHub à partir du chemin complet.

    Args:
        filepath (str): Le chemin relatif depuis la racine du dépôt.

    Returns:
        str: L'URL GitHub brute de l'image.
    """
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{filepath}"


def calculate_area(coords):
    if len(coords) < 3:
        return 0
    x, y = zip(*coords)
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(coords) - 1)))

def circle_to_coords(shape, n_points=32):
    """Renvoie une liste de coordonnées qui approchent le cercle Plotly."""
    from math import cos, sin, pi
    x0, y0, x1, y1 = shape["x0"], shape["y0"], shape["x1"], shape["y1"]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    rx = abs(x1 - x0) / 2
    ry = abs(y1 - y0) / 2
    return [
        (cx + rx * cos(2 * pi * i / n_points), cy + ry * sin(2 * pi * i / n_points))
        for i in range(n_points)
    ]

def generate_figure(image, file_val=None, shapes=None, size="normal"):
    fig = px.imshow(image)
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    if size == "mini":
        width, height = 320, 320
    else:
        width, height = 700, 700
    fig.update_layout(
        dragmode="drawclosedpath",
        uirevision=file_val or str(random.random()),
        paper_bgcolor='black',
        plot_bgcolor='black',
        width=width,
        height=height,
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=shapes if shapes is not None else [],
        newshape=dict(
            line=dict(
                color='white',
                width=2,
                dash='dot'
            )
        ),
        hovermode=False
    )
    return fig

def shape_for_plotly(shape):
    """Supprime customdata et customid avant d'envoyer à Plotly."""
    return {k: v for k, v in shape.items() if k not in ["customdata", "customid"]}

# ====================================================
# CONFIGURATION DE LA FIGURE INITIALE (AFFICHÉE EN L'ATTENTE D'UNE IMAGE)
# ====================================================
scatter_fig = go.Figure(
    go.Scattergl(
        x=np.random.randn(1000),
        y=np.random.randn(1000),
        mode='markers',
        marker=dict(
            color=random.sample(['#ecf0f1'] * 500 + ["#2d3436"] * 500, 1000),
            line_width=1
        )
    )
)
scatter_fig.update_layout(
    plot_bgcolor='#dfe6e9',
    width=700,
    height=700,
    xaxis_visible=False,
    yaxis_visible=False,
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
    hovermode=False
)

scatter_fig_mini = go.Figure(
    go.Scattergl(
        x=np.random.randn(400),
        y=np.random.randn(400),
        mode='markers',
        marker=dict(
            color=random.sample(['#ecf0f1'] * 200 + ["#2d3436"] * 200, 400),
            line_width=1
        )
    )
)
scatter_fig_mini.update_layout(
    plot_bgcolor='#dfe6e9',
    width=320,
    height=320,
    xaxis_visible=False,
    yaxis_visible=False,
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
    hovermode=False
)

config_graph = {
    "modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
    "displaylogo": False,
}

# ====================================================
# CONFIGURATION DE L'APPLICATION ET DU THÈME
# ====================================================
external_stylesheets = [
    dbc.themes.FLATLY,
    "https://use.fontawesome.com/releases/v5.15.3/css/all.css"
]

app = Dash(__name__, external_stylesheets=external_stylesheets, title="FundusTracker")
server = app.server

filenames = get_filenames()
classification_options = ["grande", "atrophie", "pigment", "incertain"]
shortcut_keys = {"grande": "g", "atrophie": "a", "pigment": "m", "incertain": "i"}

classification_buttons = [
    dbc.Button(
        opt,
        id={"type": "classify-button", "index": opt},
        color="secondary",
        style={"flex": "1", "margin": "0"},
        className="classification-button"
    )
    for opt in classification_options
]

# ====================================================
# LAYOUT DE L'APPLICATION
# ====================================================
def layout_manuelle():
    return html.Div([
        dbc.Container([
            html.Div([
                html.H2("Instructions d'utilisation"),
                html.P("1. Choisissez une image depuis le menu déroulant."),
                html.P("2. Tracez le contour du nerf optique et d'une lésion sur l'image."),
                html.P("3. Classez la zone en cliquant sur le type approprié."),
                html.H3("Vous pouvez supprimer une zone en la sélectionnant."),
                html.H3("Vous pouvez modifier une classification via le menu déroulant."),
                html.P("4. Exportez les résultats vers Excel pour obtenir un résumé."),
                html.P("5. Téléchargez les zones annotées."),
                html.H3("Vous pouvez importer un fichier avec les zones annotées."),
            ], className='left-block'),

            html.Div([
                dcc.Graph(
                    id='fig-image',
                    config=config_graph,
                    style={'width': '100%', 'height': 'auto'},
                    className="graph-figure"
                ),
                html.Div(id='output-area', className="output-area")
            ], className='middle-block'),

            html.Div([
                html.P("Choisir une image :"),
                dcc.Dropdown(
                    id='file-dropdown',
                    options=[{'label': f.split('/')[-1], 'value': f} for f in filenames], 
                    placeholder='Choisissez une image'
                ),
                html.Div("Ou chargez une image locale :", style={"marginTop": "12px"}),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Glissez-déposez une image ou ',
                        html.A('cliquez ici')
                    ]),
                    style={
                        'width': '100%',
                        'height': '40px',
                        'lineHeight': '40px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    },
                    multiple=False
                ),
                html.P("Classification :"),
                dbc.ButtonGroup(
                    classification_buttons,
                    vertical=False,
                    className="mb-2",
                    style={"width": "100%", "display": "flex"}
                ),
                dcc.Dropdown(
                    id="zone-selector",
                    options=[],
                    placeholder="Sélectionnez une zone à reclassifier"
                ),
                html.P("Réinitialiser :"),
                dbc.Button([
                    html.I(className="fas fa-undo", style={"margin-right": "5px"}),
                    "Réinitialiser les zones annotées"
                ], id="reset-button", color="danger", className="mb-2"),
                html.P("Exporter :"),
                dbc.Button([
                    html.I(className="fas fa-download", style={"margin-right": "5px"}),
                    "Exporter les résultats dans un tableur"
                ], id="export-button", color="primary", className="mb-2"),
                dcc.Download(id="download-dataframe-xlsx"),
                dbc.Button([
                    html.I(className="fas fa-file-export", style={"margin-right": "5px"}),
                    "Exporter les annotations"
                ], id="download-json-button", color="primary", className="mb-2"),
                dcc.Download(id="download-json"),

                html.P("Comparaison :"),
                dbc.Button([
                    html.I(id = "added-to-compare-feedback-left", className="fas fa-flag-checkered", style={"marginRight": "6px"}),
                    "Définir comme référence"
                ], id="add-to-compare-left", color="info", className="mb-2", style={"width": "100%"}),

                dbc.Button([
                    html.I(id = "added-to-compare-feedback-right", className="fas fa-chart-line ", style={"marginRight": "6px"}),
                    "Ajouter à la comparaison"
                ], id="add-to-compare-right", color="info", className="mb-2", style={"width": "100%"}),

                html.P("Paramètres d'affichage :"),
                dbc.FormGroup(
                    [
                        dbc.Checkbox(
                            id="show-zone-numbers",
                            checked=True,  # Coché par défaut
                            className="form-check-input"
                        ),
                        dbc.Label(
                            "Afficher le numéro des zones sur le dessin",
                            html_for="show-zone-numbers",
                            className="form-check-label"
                        )
                    ],

                    check=True,
                    className="mb-2"
                ),
                dbc.FormGroup(
                             [
                        dbc.Checkbox(
                            id="dashed-contour",
                            checked=True,  # Contour en pointillé par défaut
                            className="form-check-input"
                        ),
                        dbc.Label(
                            "Contour pointillé des formes",
                            html_for="dashed-contour",
                            className="form-check-label"
                        )
                    ],
                    check=True,
                    className="mb-2"
                ),
                html.P("Ajustements globaux :"),
                dbc.Button(
                    [
                        html.I(className="fas fa-circle", style={"margin-right": "5px"}),
                        "Ajouter le nerf optique"
                    ],
                    id="add-nerf-optique-button",
                    color="info",
                    className="mb-2",
                    style={"width": "100%"}
                ),
                html.Label("Zoom global :"),
                dcc.Slider(
                    id='zoom-slider',
                    min=0.80, max=1.2, step=0.01, value=1.0,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
                html.Label("Rotation globale (°) :"),
                dcc.Slider(
                    id='rotation-slider',
                    min=-30, max=30, step=0.5, value=0,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
                html.P("Importer :"),
                html.Div(
                    id='upload-div',
                    children=[
                        dcc.Upload(
                            id='upload-annotations',
                            children=html.Div([
                                html.I(className="fas fa-upload", style={"margin-right": "5px"}),
                                "Glissez-déposez ou sélectionnez un fichier annoté"
                            ]),
                            className="upload-area",
                            style={"width": "100%"},
                            multiple=False
                        )
                    ]
                ),
                html.Div(id='output-text', className="output-text")
            ], className='right-block')
        ],
            fluid=True,
            className='dashboard-container',
            style={'display': 'flex', 'justify-content': 'space-between'}
        ),
    ])

def layout_ml():
    return html.Div([
        # Bloc de gauche
        html.Div([
            html.H2("Instructions d'utilisation"),
            html.P("1. Choisissez une image."),
            html.P("2. Dessinez à main levée les zones d'intéret sur l’image pour donner des exemples."),
            html.P("3. Cliquez sur Segmenter pour lancer la segmentation automatique."),
            html.P("4. Vous pouvez ajouter d’autres traits et relancer la segmentation pour affiner."),
            html.P("5. Acceptez la segmentation pour travailler manuellement les zones."),
            html.Br(),
        ], className='left-block'),

        # Bloc du milieu
        html.Div([
            dcc.Graph(
                id='ml-image-graph',
                config={
                    "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],
                    "displaylogo": False,
                },
                style={'width': '100%', 'height': 'auto'},
                className="graph-figure"
            ),
            html.Div(id="ml-segment-result"),
        ], className='middle-block'),

        # Bloc de droite
        html.Div([
            html.P("Choix de l'image :"),
            dcc.Dropdown(
                id='ml-file-dropdown',
                options=[{'label': f.split('/')[-1], 'value': f} for f in filenames],
                placeholder="Choisissez une image"
            ),
            html.P("Pinceau/étiquette :"),
            dcc.Dropdown(
                id="ml-label-dropdown",
                options=[
                    {'label': "Papille (jaune)", 'value': 1},
                    {'label': "Lésion (rouge)", 'value': 2},
                    {'label': "Fond (vert)", 'value': 3},
                ],
                value=1,
                style={"width": "100%"}
            ),
            html.P("Taille du pinceau :"),
            dcc.Slider(
                id="ml-line-width",
                min=1, max=20, step=1, value=7,
                tooltip={"placement": "bottom", "always_visible": False}
            ),
            html.P("Segmentation :"),
            dbc.Button([
                html.I(className="fas fa-magic", style={"margin-right": "5px"}),
                "Segmenter les zones dessinées"
            ], id="ml-segment-btn", color="primary", className="mb-2"),
            dbc.Button([
                html.I(className="fas fa-check", style={"marginRight": "5px"}),
                "Accepter la segmentation"
            ], id="ml-accept-zones-btn", color="success", className="mb-2", style={"width": "100%"}),
            html.P("Réinitialisation"),
            dbc.Button([
                html.I(className="fas fa-undo", style={"marginRight": "5px"}),
                "Réinitialiser la segmentation"
            ], id="ml-reset-btn", color="danger", className="mb-2", style={"width": "100%"}),
        ], className='right-block'),
    ],
    className='dashboard-container',
    style={'display': 'flex', 'justify-content': 'space-between'}
    )

def layout_compare():
    return html.Div([
        # Bloc à gauche : infos patient
        html.Div([
            html.H2("Informations du patient"),
            dbc.FormGroup([
                dbc.Label("Nom du patient :"),
                dbc.Input(id="patient-nom", type="text", placeholder="Nom"),
            ]),
            dbc.FormGroup([
                dbc.Label("Prénom du patient :"),
                dbc.Input(id="patient-prenom", type="text", placeholder="Prénom"),
            ]),
            dbc.FormGroup([
                dbc.Label("Date de naissance :"),
                dbc.Input(id="patient-dob", type="date"),
            ]),
            dbc.FormGroup([
                dbc.Label("Date image de référence :"),
                dbc.Input(id="date-gauche", type="date"),
            ]),
            dbc.FormGroup([
                dbc.Label("Date image de comparaison :"),
                dbc.Input(id="date-droite", type="date"),
            ]),
            dbc.Button(
                [
                    html.I(className="fas fa-save", style={"marginRight": "8px"}),
                    "Enregistrer infos patient"
                ],
                id="save-patient-infos-btn",
                color="primary",
                className="mb-3",
                style={"marginTop": "10px", "width": "100%", "display": "flex", "alignItems": "center",
                       "justifyContent": "center"}
            ),
            html.Div(id="save-patient-infos-feedback"),
        ], className="left-block"),

        # Bloc à droite : comparatif images + synthèse
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id="compare-fig-left", style={"width": "100%", "height": "320px"}),
                    html.P(id="compare-image-name-left", style={"textAlign": "center", "fontSize": "13px"}),
                ], style={"flex": "1", "marginRight": "10px"}),
                html.Div([
                    dcc.Graph(id="compare-fig-right", style={"width": "100%", "height": "320px"}),
                    html.P(id="compare-image-name-right", style={"textAlign": "center", "fontSize": "13px"}),
                ], style={"flex": "1", "marginLeft": "10px"}),
            ], style={"display": "flex", "flexDirection": "row", "marginBottom": "20px", "width": "100%"}),

            # Synthèse/rapport en-dessous
            html.Div(id="evolution-summary", style={"width": "100%"}),
            dbc.Button(
                [
                    html.I(className="fas fa-file-pdf", style={"marginRight": "6px"}),
                    "Exporter le rapport"
                ],
                id="export-report-btn",
                color="secondary",
                style={"margin": "12px 0"},
                n_clicks=0,
            )
        ], className="right-block", style={"flex": "2"}),
        html.Div(id="pdf-report-content", style={"display": "none"})
    ],
    className="dashboard-container",
    style={'display': 'flex', 'justifyContent': 'space-between'})

app.layout = html.Div([
    # Logo placé au-dessus des onglets
    html.Div(
        children=[
            html.Img(
                src=app.get_asset_url('logo.png'),
                style={'height': '100px', 'verticalAlign': 'middle', 'marginRight': '10px'}
            ),
            html.Span(
                "FundusTracker",
                style={"fontSize": "40px", "verticalAlign": "middle", "fontWeight": "bold"}
            )
        ],
        className="logo-container",
        style={"textAlign": "center"}
    ),

    # Les onglets (tabs)
    dcc.Tabs(
        id="tabs",
        value="tab-ml",
        children=[
            dcc.Tab(label="Segmentation semi-automatique (BETA)", value="tab-ml", children=layout_ml()),
            dcc.Tab(label="Segmentation manuelle", value="tab-manuelle", children=layout_manuelle()),
            dcc.Tab(label="Comparaison", value="tab-compare", children=layout_compare())
        ],
    ),
html.Footer(
            html.Div([
                "© 2025 – Réalisé par ",
                html.A(
                    "Thomas Foulonneau",
                    href="https://www.linkedin.com/in/thomas-foulonneau?originalSubdomain=fr",
                    target="_blank",
                    style={"color": "#ffffff", "textDecoration": "underline"}
                ),
                " – Interne à l'Ophtalmopole de Paris"
            ]),
            className="footer"
        ),
    # Stores globaux en bas pour une organisation claire
    dcc.Store(id="patient-infos-store", data={}),
    dcc.Store(id="compare-shapes-left", data=None),
    dcc.Store(id="compare-shapes-right", data=None),
    dcc.Store(id="compare-image-left", data=None),
    dcc.Store(id="compare-image-right", data=None),
    dcc.Store(id="stored-shapes", data=[]),
    dcc.Store(id="ml-squiggle-store", data=[]),
    dcc.Store(id="ml-segmentation-mask", data=None),
    dcc.Store(id="tab-value-store", data="tab-manuelle"),
    dcc.Store(id="trigger-print-store", data=False),
    dcc.Store(id='uploaded-image-store', data=None),

])


# ====================================================
# CALLBACK 1 : MISE À JOUR DE LA FIGURE D'AFFICHAGE
# ====================================================
@app.callback(
    Output("fig-image", "figure"),
    Input("file-dropdown", "value"),
    Input("uploaded-image-store", "data"),
    Input("reset-button", "n_clicks"),
    Input("stored-shapes", "data"),
    Input("show-zone-numbers", "checked"),
    Input("dashed-contour", "checked"),
    Input("zoom-slider", "value"),
    Input("rotation-slider", "value"),
    State("fig-image", "figure")
)
def update_figure(file_val, uploaded_image, reset_clicks, stored_shapes, show_zone_numbers, dashed_contour, zoom, rotation, current_fig):
    image_id = file_val if file_val else uploaded_image
    if image_id:
        img = load_image_any(image_id)
        fig = generate_figure(img, file_val=image_id)
        width, height = img.size
    else:
        fig = scatter_fig
        width, height = 700, 700

    cx, cy = width / 2, height / 2

    # 2. Affiche les shapes avec transformation
    if stored_shapes is not None:
        plotly_shapes = []
        for shape in stored_shapes:
            shape_t = transform_shape(shape, zoom, rotation, (cx, cy))
            shape_t.setdefault("editable", True)
            shape_t.setdefault("layer", "above")
            shape_t.setdefault("xref", "x")
            shape_t.setdefault("yref", "y")
            shape_t.setdefault("line", {"width": 0.1})
            shape_t["line"]["dash"] = "dot" if dashed_contour else "solid"
            plotly_shapes.append(shape_for_plotly(shape_t))
        fig["layout"]["shapes"] = plotly_shapes

        def centroid(coords):
            if not coords:
                return 0, 0
            avg_x = sum(x for x, y in coords) / len(coords)
            avg_y = sum(y for x, y in coords) / len(coords)
            return avg_x, avg_y

        annotations = []
        for i, shape in enumerate(stored_shapes):
            if shape.get("type") == "circle":
                coords = circle_to_coords(shape)
            else:
                path_str = shape.get("path", "")
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
                try:
                    coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
                except Exception:
                    coords = []
            coords_t = transform_coords(coords, zoom, rotation, (cx, cy))
            cx_ann, cy_ann = centroid(coords_t)
            annotations.append(dict(
                x=cx_ann,
                y=cy_ann,
                text=str(i + 1),
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
                font=dict(color="white", size=12)
            ))
        fig["layout"]["annotations"] = annotations if show_zone_numbers else []

    return fig



# ====================================================
# CALLBACK 2 : GESTION DES ANNOTATIONS, CLASSIFICATIONS, RÉINITIALISATION ET UPLOAD
# ====================================================
@app.callback(
    Output("stored-shapes", "data"),
    Output("output-area", "children"),
    Output("upload-div", "children"),
    Output("file-dropdown", "value"),    # <-- Pour forcer l'image affichée côté manuel
    Output("tab-value-store", "data"),   # <-- Pour forcer le switch d'onglet
    Input("add-nerf-optique-button", "n_clicks"),
    Input("fig-image", "relayoutData"),
    Input("reset-button", "n_clicks"),
    Input({"type": "classify-button", "index": ALL}, "n_clicks"),
    Input("upload-annotations", "contents"),
    Input("file-dropdown", "value"),
    Input("ml-accept-zones-btn", "n_clicks"),
    State("stored-shapes", "data"),
    State("zone-selector", "value"),
    State("ml-segmentation-mask", "data"),
    State("ml-file-dropdown", "value"),    # On ajoute le nom du fichier ML comme state
    prevent_initial_call=True
)
def update_shapes_combined(
    add_nerf_clicks,
    relayout_data,
    reset_clicks,
    classify_clicks,
    upload_contents,
    file_val,
    ml_accept_clicks,
    stored_shapes,
    selected_zone,
    mask_json,
    ml_file_val
):
    import numpy as np
    import json
    trigger = ctx.triggered_id
    new_upload = dash.no_update

    if stored_shapes is None:
        stored_shapes = []

    # ----- GESTION DU ML (Acceptation des zones auto) -----
    if trigger == "ml-accept-zones-btn":
        if not mask_json or not ml_file_val:
            return dash.no_update, "Aucune segmentation ML détectée.", new_upload, dash.no_update, dash.no_update
        mask = np.array(json.loads(mask_json))

        # Génère la papille (label 1) et les lésions (label 2)
        papille_shapes = mask_to_shapes(mask, label_value=1, min_area=20)  # papille: zone plus petite OK
        lesion_shapes = mask_to_shapes(mask, label_value=2, min_area=200)

        shapes = []
        id_counter = 1

        # Ajoute la papille si présente (comme zone 1, blanche, pointillé, customid=1, customdata="nerf optique")
        if papille_shapes:
            papille_shape = papille_shapes[0]
            papille_shape["customid"] = 1
            papille_shape["customdata"] = "nerf optique"
            papille_shape["line"] = {"color": "white", "width": 2, "dash": "dot"}
            papille_shape["layer"] = "above"
            shapes.append(papille_shape)
            id_counter = 2

        # Ajoute les lésions ensuite (customid croissant, customdata=segmentation-auto)
        for i, sh in enumerate(lesion_shapes):
            sh["customid"] = id_counter
            sh["customdata"] = "segmentation-auto"
            sh["line"] = {"color": "yellow", "width": 2, "dash": "solid"}
            sh["layer"] = "above"
            shapes.append(sh)
            id_counter += 1

        summary = générer_resume(shapes)
        return shapes, summary, new_upload, ml_file_val, "tab-manuelle"

    # ----- RESET -----
    if trigger == "reset-button":
        new_upload = [
            dcc.Upload(
                id='upload-annotations',
                children=html.Div([
                    'Glissez-déposez ou ',
                    html.A('sélectionnez un fichier annoté', className="upload-link")
                ]),
                className="upload-area",
                multiple=False
            )
        ]
        return [], "Annotations réinitialisées.", new_upload, dash.no_update, dash.no_update

    # ----- IMPORT JSON -----
    if trigger == "upload-annotations" and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            new_annotations = json.loads(decoded.decode('utf-8'))
        except Exception:
            new_annotations = []
        shapes = []
        for i, shape in enumerate(new_annotations):
            if "customid" not in shape:
                shape["customid"] = i + 1
            shapes.append(shape)
        summary = générer_resume(shapes)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    # ----- OUVERTURE D’UNE IMAGE -----
    if trigger == "file-dropdown" and file_val:
        if not stored_shapes or len(stored_shapes) == 0:
            try:
                img = load_image_any(file_val)
                width, height = img.size
            except Exception:
                width, height = 700, 700
            cx, cy = width / 2, height / 2
            r = 50  # rayon par défaut
            cercle_nerf = {
                "type": "circle",
                "xref": "x", "yref": "y",
                "x0": cx - r, "y0": cy - r,
                "x1": cx + r, "y1": cy + r,
                "line": {"color": "white", "width": 2, "dash": "dot"},
                "customdata": "nerf optique",
                "customid": 1,
                "editable": True,
                "layer": "above"
            }
            shapes = [cercle_nerf]
        else:
            shapes = stored_shapes
        summary = générer_resume(shapes)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    # ----- AJOUT DU NERF OPTIQUE -----
    if trigger == "add-nerf-optique-button":
        image_id = file_val if file_val else None
        if image_id:
            try:
                img = load_image_any(image_id)
                width, height = img.size
            except Exception:
                width, height = 700, 700
        else:
            width, height = 700, 700
        cx, cy = width / 2, height / 2
        r = 50
        already_nerf = any(
            (s.get("customdata", "") == "nerf optique" or s.get("customid", 0) == 1)
            for s in stored_shapes
        )
        if not already_nerf:
            cercle_nerf = {
                "type": "circle",
                "xref": "x", "yref": "y",
                "x0": cx - r, "y0": cy - r,
                "x1": cx + r, "y1": cy + r,
                "line": {"color": "white", "width": 2, "dash": "dot"},
                "customdata": "nerf optique",
                "customid": 1,
                "editable": True,
                "layer": "above"
            }
            shapes = [cercle_nerf] + stored_shapes
        else:
            shapes = stored_shapes
        summary = générer_resume(shapes)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    # ----- CLASSIFICATION -----
    if isinstance(trigger, dict) and trigger.get("type") == "classify-button":
        label = trigger["index"]
        shapes = stored_shapes.copy()
        if selected_zone is not None and selected_zone < len(shapes):
            shapes[selected_zone]["customdata"] = label
        elif shapes:
            shapes[-1]["customdata"] = label
        summary = générer_resume(shapes)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    # ----- TRAÇAGE OU MODIF DE SHAPE -----
    if relayout_data:
        shapes = stored_shapes.copy()
        if "shapes" in relayout_data:
            new_shapes = relayout_data["shapes"]
            updated_shapes = []
            for i, new_shape in enumerate(new_shapes):
                valid = {k: v for k, v in new_shape.items() if k not in ["customdata", "customid"]}
                valid["customdata"] = shapes[i].get("customdata", "Tache") if i < len(shapes) else "Tache"
                if "customid" not in valid:
                    valid["customid"] = len(shapes) + 1
                updated_shapes.append(valid)
            shapes = updated_shapes
        else:
            import re
            for key, val in relayout_data.items():
                m = re.match(r"shapes\[(\d+)\]\.(\w+)", key)
                if m:
                    idx, prop = int(m.group(1)), m.group(2)
                    if idx < len(shapes):
                        shapes[idx][prop] = val
        summary = générer_resume(shapes)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    # ----- PAR DÉFAUT -----
    summary = générer_resume(stored_shapes)
    return stored_shapes, summary, new_upload, dash.no_update, dash.no_update

def générer_resume(shapes):
    areas = []
    for i, shape in enumerate(shapes):
        lab = shape.get("customdata", "Tache")
        if shape.get("type") == "circle":
            coords = circle_to_coords(shape)
        else:
            path_str = shape.get("path", "")
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
            try:
                coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
            except Exception:
                coords = []
        area = calculate_area(coords) if coords else 0
        areas.append(f"Zone {i + 1} : {area:.2f} pixels² ({lab})")
    return dbc.Card(
        [
            dbc.CardHeader("Résumé des zones annotées :"),
            dbc.CardBody(
                html.Ul([html.Li(a) for a in areas]),
                style={"padding": "10px"}
            )
        ],
        style={
            "marginTop": "10px",
            "border": "1px solid #cccccc",
            "borderRadius": "5px",
            "backgroundColor": "#f8f9fa"
        }
    )

# ====================================================
# CALLBACK 3 : MISE À JOUR DU DROPDOWN DE SÉLECTION DE ZONES
# ====================================================
@app.callback(
    Output("zone-selector", "options"),
    Input("stored-shapes", "data")
)
def update_zone_selector_options(stored_shapes):
    if stored_shapes is None:
        return []
    return [{"label": f"Zone {i + 1}", "value": i} for i in range(len(stored_shapes))]

# ====================================================
# CALLBACK 4 : EXPORT VERS FICHIER EXCEL
# ====================================================
@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("export-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    prevent_initial_call=True
)
def export_to_excel(n_clicks, stored_shapes, file_val):
    if not n_clicks or not stored_shapes or not file_val:
        return dash.no_update

    import numpy as np

    # Chargement de l'image pour en extraire dynamiquement le centre.
    try:
        image_id = file_val  # Ou: image_id = file_val if file_val else uploaded_image
        image = load_image_any(image_id)
        width, height = image.size
        nerf_optique_centroid = (width / 2, height / 2)
    except Exception as e:
        nerf_optique_centroid = (350, 350)

    def calc_centroid(coords):
        arr = np.array(coords)
        if len(arr) == 0:
            return None, None
        return np.mean(arr, axis=0)

    def compute_ellipse_params(coords):
        arr = np.array(coords)
        centroid = np.mean(arr, axis=0)
        cov = np.cov(arr, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eig(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]
        major_axis = 2 * np.sqrt(eigenvals[0])
        minor_axis = 2 * np.sqrt(eigenvals[1])
        ellipse_angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        return centroid, major_axis, minor_axis, ellipse_angle

    rows = []
    for i, shape in enumerate(stored_shapes):
        path_str = shape.get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
            area = calculate_area(coords)
        except Exception as e:
            area = None
            coords = []
        cx, cy = calc_centroid(coords) if coords else (None, None)
        classification = shape.get("customdata", "Tache")
        try:
            if len(coords) >= 2:
                centroid, major_axis, minor_axis, ellipse_angle = compute_ellipse_params(coords)
            else:
                major_axis = None
                minor_axis = None
                ellipse_angle = None
        except Exception as e:
            major_axis = None
            minor_axis = None
            ellipse_angle = None

        if cx is not None and cy is not None:
            # Calcul de l'angle entre le centre de l'image (nerf optique) et le centroïde de la zone
            angle_from_center = np.degrees(np.arctan2(cy - nerf_optique_centroid[1], cx - nerf_optique_centroid[0]))
        else:
            angle_from_center = None

        rows.append({
            "Zone": i + 1,
            "Aire (pixels²)": area,
            "Centroid X": cx,
            "Centroid Y": cy,
            "Classification": classification,
            "Grand Axe (pixels)": major_axis,
            "Petit Axe (pixels)": minor_axis,
            "Angle (degrés) par rapport Nerf Optique": angle_from_center
        })

    df = pd.DataFrame(rows)
    filename = f"{file_val.split('.')[0]}.xlsx" if file_val else "export.xlsx"

    def to_excel(bytes_io):
        with pd.ExcelWriter(bytes_io, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Zones")

    return dcc.send_bytes(to_excel, filename)


# ====================================================
# CALLBACK 5 : EXPORT DES ANNOTATIONS EN JSON
# ====================================================
@app.callback(
    Output("download-json", "data"),
    Input("download-json-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    prevent_initial_call=True
)
def download_annotations(n_clicks, stored_shapes, file_val):
    if not stored_shapes:
        return dash.no_update
    content = json.dumps(stored_shapes)
    filename = f"{file_val.split('.')[0]}.json" if file_val else "annotations.json"
    return dcc.send_string(content, filename)


@app.callback(
    Output("ml-image-graph", "figure"),
    Output("ml-squiggle-store", "data"),
    Input("ml-file-dropdown", "value"),
    Input("ml-image-graph", "relayoutData"),
    Input("ml-label-dropdown", "value"),
    Input("ml-line-width", "value"),
    Input("ml-reset-btn", "n_clicks"),
    State("ml-squiggle-store", "data"),
)
def update_ml_figure(file_val, relayout_data, label, width, ml_reset_clicks, squiggles):
    trigger = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    if squiggles is None:
        squiggles = []

    if trigger == "ml-reset-btn":
        if file_val:
            img = load_image_any(file_val)
            arr = np.array(img)
            fig = px.imshow(arr)
            fig.update_xaxes(showticklabels=False, visible=False)
            fig.update_yaxes(showticklabels=False, visible=False)
            fig.update_layout(
                dragmode="drawopenpath",
                width=700, height=700,
                margin=dict(l=0, r=0, t=0, b=0),
                shapes=[],
                uirevision=str(random.random())
            )
        else:
            fig = scatter_fig
        return fig, []

    if file_val:
        img = load_image_any(file_val)
        arr = np.array(img)
        fig = px.imshow(arr)
        fig.update_xaxes(showticklabels=False, visible=False)
        fig.update_yaxes(showticklabels=False, visible=False)
        fig.update_layout(
            dragmode="drawopenpath",
            width=700, height=700,
            margin=dict(l=0, r=0, t=0, b=0),
            shapes=[],
            uirevision=file_val
        )
    else:
        fig = scatter_fig

    shapes = []
    colors = {1: "yellow", 2: "red", 3: "lime"}
    if squiggles:
        for squig in squiggles:
            shapes.append({
                "type": "path",
                "path": squig["path"],
                "line": {"color": colors.get(squig["label"], "yellow"), "width": squig.get("width", 7)},
                "layer": "above"
            })

    if relayout_data and "shapes" in relayout_data:
        for i, sh in enumerate(relayout_data["shapes"]):
            if i >= len(squiggles):
                squiggles.append({
                    "path": sh["path"],
                    "label": label,
                    "width": width,
                })
        shapes = []
        for squig in squiggles:
            shapes.append({
                "type": "path",
                "path": squig["path"],
                "line": {"color": colors.get(squig["label"], "yellow"), "width": squig.get("width", 7)},
                "layer": "above"
            })

    fig.update_layout(shapes=shapes)
    return fig, squiggles


@app.callback(
    Output("ml-segment-result", "children"),
    Output("ml-image-graph", "figure", allow_duplicate=True),
    Output("ml-segmentation-mask", "data"),
    Input("ml-segment-btn", "n_clicks"),
    Input("ml-reset-btn", "n_clicks"),
    State("ml-file-dropdown", "value"),
    State("ml-squiggle-store", "data"),
    prevent_initial_call=True
)
def ml_run_segmentation(n_seg, n_reset, file_val, squiggles):
    triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    if triggered == "ml-reset-btn":
        return "", dash.no_update, None

    if not file_val or not squiggles or len(squiggles) < 2:
        return "Ajoutez au moins 2 squiggles (fond + lésion).", dash.no_update, dash.no_update

    image = load_image_any(file_val)
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr_gray = color.rgb2gray(arr) if arr.ndim == 3 else arr
    h, w = arr_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for squig in squiggles:
        pts = re.findall(r'[-+]?\d*\.\d+|\d+', squig["path"])
        pts = np.array([[float(pts[i]), float(pts[i+1])] for i in range(0, len(pts), 2)]).astype(int)
        for x, y in pts:
            for dx in range(-squig["width"]//2, squig["width"]//2+1):
                for dy in range(-squig["width"]//2, squig["width"]//2+1):
                    xi, yi = int(x+dx), int(y+dy)
                    if 0 <= xi < w and 0 <= yi < h:
                        mask[yi, xi] = squig["label"]
    idx = mask > 0
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    H_elems = hessian_matrix(arr_gray, sigma=2, order='rc')
    eigvals = hessian_matrix_eigvals(H_elems)
    eigval0 = eigvals[0].ravel()
    features = np.stack([
        arr_gray.ravel(),
        filters.gaussian(arr_gray, sigma=1).ravel(),
        filters.sobel(arr_gray).ravel(),
        eigval0,
        Y.ravel() / h, X.ravel() / w
    ], axis=1)
    X_train = features[idx.ravel()]
    y_train = mask[idx]
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=25)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(features)
    mask_pred = y_pred.reshape(h, w)
    fig = px.imshow(arr)
    fig.add_trace(go.Heatmap(
        z=mask_pred,
        showscale=False,
        opacity=0.45,
        colorscale=[[0, "rgba(0,0,0,0)"], [0.5, "rgba(255,0,0,0.5)"], [1, "rgba(0,255,0,0.5)"]],
    ))
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)
    fig.update_layout(width=700, height=700, margin=dict(l=0, r=0, t=0, b=0))
    mask_json = json.dumps(mask_pred.tolist())
    return html.Div("Segmentation calculée ! Cliquez sur « Accepter comme zones » pour exporter les lésions."), fig, mask_json


@app.callback(
    Output("tabs", "value"),
    Input("tab-value-store", "data"),
    prevent_initial_call=True
)
def update_tabs(tab_value):
    return tab_value

# Ajout à la comparaison gauche
@app.callback(
    Output("compare-shapes-left", "data"),
    Output("compare-image-left", "data"),
    Output("added-to-compare-feedback-left", "children"),
    Input("add-to-compare-left", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    State("uploaded-image-store", "data"),   # Ajoute ici !
    prevent_initial_call=True
)
def add_left(n, shapes, file_val, uploaded_image):
    if not n or not shapes: return dash.no_update, dash.no_update, ""
    # Si pas de fichier sélectionné mais image uploadée, stocke la base64
    if file_val:
        image_id = file_val
    elif uploaded_image:
        image_id = uploaded_image
    else:
        image_id = None
    return shapes, image_id, dbc.Alert("Ajouté comme image de référence !", color="success", duration=1500)


@app.callback(
    Output("compare-shapes-right", "data"),
    Output("compare-image-right", "data"),
    Output("added-to-compare-feedback-right", "children"),
    Input("add-to-compare-right", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    State("uploaded-image-store", "data"),  # Ajout ici !
    prevent_initial_call=True
)
def add_right(n, shapes, file_val, uploaded_image):
    if not n or not shapes:
        return dash.no_update, dash.no_update, ""
    # Si un fichier GitHub est sélectionné, on le prend en priorité
    if file_val:
        image_id = file_val
    # Sinon, si une image locale a été uploadée, on la prend
    elif uploaded_image:
        image_id = uploaded_image
    else:
        image_id = None
    return shapes, image_id, dbc.Alert("Ajouté comme image de comparaison !", color="success", duration=1500)

@app.callback(
    Output("compare-fig-left", "figure"),
    Output("compare-fig-right", "figure"),
    Output("compare-image-name-left", "children"),
    Output("compare-image-name-right", "children"),
    Output("evolution-summary", "children"),
    Output("pdf-report-content", "children"),
    Input("compare-shapes-left", "data"),
    Input("compare-shapes-right", "data"),
    Input("compare-image-left", "data"),
    Input("compare-image-right", "data"),
    Input("patient-infos-store", "data"),
)
def update_comparison(shapes_left, shapes_right, img_left, img_right, patient_infos):
    # LEFT
    if img_left:
        img = load_image_any(img_left)
        fig_left = generate_figure(img, shapes=[shape_for_plotly(s) for s in shapes_left] if shapes_left else [], size="mini")
    else:
        fig_left = scatter_fig_mini

    # RIGHT
    if img_right:
        img = load_image_any(img_right)
        fig_right = generate_figure(img, shapes=[shape_for_plotly(s) for s in shapes_right] if shapes_right else [], size="mini")
    else:
        fig_right = scatter_fig_mini

    date_gauche = patient_infos.get("date_gauche")
    date_droite = patient_infos.get("date_droite")
    name_left = f"Image de référence  : {date_gauche}" if date_gauche else "Image de référence"
    name_right = f"Image de comparaison : {date_droite}" if date_droite else "Image de comparaison"
    summary = generate_comparison_summary(shapes_left, shapes_right, patient_infos)
    pdf_report_div = render_pdf_report(patient_infos, summary)
    return fig_left, fig_right, name_left, name_right, summary, pdf_report_div

from scipy.stats import ttest_ind


def aire_nerf_optique(shapes):
    # Cherche la shape du nerf optique, si pas trouvé -> retourne None
    for shape in shapes:
        if shape.get("customdata") == "nerf optique" or shape.get("customid", 0) == 1:
            if shape.get("type") == "circle":
                coords = circle_to_coords(shape)
            else:
                path_str = shape.get("path", "")
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
                try:
                    coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
                except Exception:
                    coords = []
            return calculate_area(coords)
    return None


def aires_lesions(shapes):
    areas = []
    for shape in shapes:
        if shape.get("customdata", "").lower() not in ["nerf optique"]:
            if shape.get("type") == "circle":
                coords = circle_to_coords(shape)
            else:
                path_str = shape.get("path", "")
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
                try:
                    coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
                except Exception:
                    coords = []
            if coords:
                area = calculate_area(coords)
                areas.append(area)
    return areas

from datetime import datetime

def generate_comparison_summary(shapes_left, shapes_right, patient_infos=None):
    if not shapes_left or not shapes_right:
        return dbc.Alert("Les deux images doivent être ajoutées à la comparaison.", color="warning")
    area_nerf_left = aire_nerf_optique(shapes_left)
    area_nerf_right = aire_nerf_optique(shapes_right)
    lesion_areas_left = aires_lesions(shapes_left)
    lesion_areas_right = aires_lesions(shapes_right)
    total_left = sum(lesion_areas_left)
    total_right = sum(lesion_areas_right)
    pct_left = (total_left / area_nerf_left * 100) if area_nerf_left else None
    pct_right = (total_right / area_nerf_right * 100) if area_nerf_right else None
    diff = total_right - total_left

    # Test stat (inchangé)
    if len(lesion_areas_left) > 1 and len(lesion_areas_right) > 1:
        var_left = np.var(lesion_areas_left, ddof=1)
        var_right = np.var(lesion_areas_right, ddof=1)
        se_diff = np.sqrt(var_left/len(lesion_areas_left) + var_right/len(lesion_areas_right))
        if se_diff > 0:
            z_score = diff / se_diff
            p_value_sum = 2 * (1 - norm.cdf(abs(z_score)))
        else:
            p_value_sum = None
    else:
        p_value_sum = None

    # === Croissance temporelle ===
    growth_txt = ""
    if patient_infos:
        date_gauche = patient_infos.get("date_gauche")
        date_droite = patient_infos.get("date_droite")
        if date_gauche and date_droite:
            try:
                d0 = datetime.strptime(date_gauche, "%Y-%m-%d")
                d1 = datetime.strptime(date_droite, "%Y-%m-%d")
                days = (d1 - d0).days
                if days > 0:
                    n_months = days / 30.44
                    n_years = days / 365.25
                    growth_month = diff / n_months if n_months != 0 else None
                    growth_year = diff / n_years if n_years != 0 else None
                    pct_month = (growth_month / total_left * 100) if total_left and growth_month is not None else None
                    pct_year = (growth_year / total_left * 100) if total_left and growth_year is not None else None

                    # Affichage adapté à la durée
                    if n_years < 1:
                        growth_txt = (
                            f"**Variation d’aire totale :** {growth_month:.1f} pixels²/mois  \n"
                            f"**Variation relative :** {pct_month:.2f}%/mois"
                        ) if growth_month is not None else ""
                    else:
                        growth_txt = (
                            f"**Variation d’aire totale :** {growth_year:.1f} pixels²/an  \n"
                            f"**Variation relative :** {pct_year:.2f}%/an"
                        ) if growth_year is not None else ""
            except Exception:
                pass

    table = dbc.Table([
        html.Thead(html.Tr([html.Th(""), html.Th("T0"), html.Th("Tx")])),
        html.Tbody([
            html.Tr([html.Td("Nombre de taches"), html.Td(len(lesion_areas_left)), html.Td(len(lesion_areas_right))]),
            html.Tr([
                html.Td("Aire totale (pixels²)"),
                html.Td(f"{total_left:.0f}"),
                html.Td(f"{total_right:.0f}")
            ]),
            html.Tr([
                html.Td("Aire totale (% du nerf optique)"),
                html.Td(f"{pct_left:.1f}%" if pct_left is not None else "Non calculé"),
                html.Td(f"{pct_right:.1f}%" if pct_right is not None else "Non calculé")
            ]),
            html.Tr([
                html.Td("Différence brute (Tx-T0)"),
                html.Td(colSpan=2, children=f"{diff:.0f} pixels²")
            ]),
            html.Tr([
                html.Td("p.value (diff. aire totale)"),
                html.Td(colSpan=2, children=f"{p_value_sum:.3g}" if p_value_sum is not None else "Non calculable")
            ]),
        ])
    ], bordered=True, striped=True, hover=True, size="sm")
    mean_left = np.mean(lesion_areas_left) if len(lesion_areas_left) > 0 else 0
    mean_right = np.mean(lesion_areas_right) if len(lesion_areas_right) > 0 else 0
    date_gauche = patient_infos.get("date_gauche", "T0") if patient_infos else "T0"
    date_droite = patient_infos.get("date_droite", "Tx") if patient_infos else "Tx"

    fig = bar_dot_plot_evolution(
        total_left, total_right, lesion_areas_left, lesion_areas_right, mean_left, mean_right,
        date_left=date_gauche, date_right=date_droite
    )

    # Affiche la croissance si disponible
    growth_html = html.Div([
        html.Hr(style={"margin": "8px 0"}),
        dcc.Markdown(growth_txt)
    ]) if growth_txt else None

    return dbc.Card([
        dbc.CardHeader("Synthèse de l'évolution"),
        dbc.CardBody([
            table,
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "240px", "marginBottom": "6px"}),
            growth_html if growth_html else None
        ])
    ])
import plotly.graph_objects as go
def bar_dot_plot_evolution(total_left, total_right, lesion_areas_left, lesion_areas_right, mean_left, mean_right, date_left="T0", date_right="Tx"):
    labels = [str(date_left or "T0"), str(date_right or "Tx")]

    # Always as lists/arrays
    lesion_areas_left = lesion_areas_left if isinstance(lesion_areas_left, (list, np.ndarray)) else []
    lesion_areas_right = lesion_areas_right if isinstance(lesion_areas_right, (list, np.ndarray)) else []

    fig = go.Figure()
    # 1. Barres (aire totale)
    fig.add_trace(go.Bar(
        x=labels,
        y=[total_left, total_right],
        name="Aire totale",
        marker_color="#1976D2",
        width=0.5,
        opacity=0.25
    ))
    # 2. Dotplot : tous les points (aires des lésions individuelles)
    fig.add_trace(go.Scatter(
        x=[labels[0]]*len(lesion_areas_left),
        y=lesion_areas_left,
        mode='markers',
        name='Aires lésionnelles T0',
        marker=dict(size=9, color="#1abc9c", opacity=0.6, line=dict(width=1, color="white")),
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[labels[1]]*len(lesion_areas_right),
        y=lesion_areas_right,
        mode='markers',
        name='Aires lésionnelles Tx',
        marker=dict(size=9, color="#e67e22", opacity=0.6, line=dict(width=1, color="white")),
        showlegend=True
    ))
    # 3. Moyenne
    fig.add_trace(go.Scatter(
        x=labels,
        y=[mean_left, mean_right],
        mode="lines+markers",
        name="Moyenne",
        marker=dict(size=16, symbol="diamond", color="#e74c3c"),
        line=dict(width=2, dash="dot", color="#e74c3c"),
        showlegend=True,
        hovertemplate="Moyenne : %{y:.0f} pixels²"
    ))
    fig.update_layout(
        yaxis_title="Aire (pixels²)",
        xaxis_title="Date",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        height=260,
        margin=dict(l=12, r=12, t=38, b=10),
        xaxis = dict(type="category")
    )
    return fig




@app.callback(
    Output("patient-infos-store", "data"),
    Output("save-patient-infos-feedback", "children"),
    Input("save-patient-infos-btn", "n_clicks"),
    State("patient-nom", "value"),
    State("patient-prenom", "value"),
    State("patient-dob", "value"),
    State("date-gauche", "value"),
    State("date-droite", "value"),
    prevent_initial_call=True
)
def save_patient_infos(n, nom, prenom, dob, date_gauche, date_droite):
    if not n:
        return dash.no_update, ""
    infos = {
        "nom": nom,
        "prenom": prenom,
        "dob": dob,
        "date_gauche": date_gauche,
        "date_droite": date_droite,
        "last_save": datetime.now().isoformat()
    }
    feedback = dbc.Alert("Informations patient enregistrées !", color="success", duration=2000)
    return infos, feedback

@app.callback(
    Output("trigger-print-store", "data"),
    Input("export-report-btn", "n_clicks"),
    prevent_initial_call=True
)
def trigger_pdf_export(n):
    if n:
        return True
    return dash.no_update


def render_pdf_report(patient_infos, summary_card):
    nom = patient_infos.get("nom", "") if patient_infos else ""
    prenom = patient_infos.get("prenom", "") if patient_infos else ""
    dob = patient_infos.get("dob", "") if patient_infos else ""

    return html.Div(
        [
            # Header (logo + titre)
            html.Div([
                html.Img(
                    src="/assets/logo.png",  # OU app.get_asset_url("logo.png")
                    style={"height": "60px", "verticalAlign": "middle", "marginRight": "10px"}
                ),
                html.Span(
                    "FundusTracker",
                    style={"fontSize": "28px", "verticalAlign": "middle", "fontWeight": "bold"}
                )
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
            html.Hr(),

            # Infos patient
            html.Div([
                html.P(f"Nom : {nom}"),
                html.P(f"Prénom : {prenom}"),
                html.P(f"Date de naissance : {dob}"),
            ], style={"marginBottom": "14px", "fontSize": "16px"}),

            # Synthèse (summary_card)
            summary_card,
            html.Hr(),

            # Footer
            html.Footer(
                html.Div([
                    "© 2025 – Réalisé par ",
                    html.A(
                        "Thomas Foulonneau",
                        href="https://www.linkedin.com/in/thomas-foulonneau?originalSubdomain=fr",
                        target="_blank",
                        style={"color": "#636e72", "textDecoration": "underline"}
                    ),
                    " – Interne à l'Ophtalmopole de Paris"
                ]),
                style={
                    "fontSize": "12px", "textAlign": "center", "color": "#636e72",
                    "background": "none", "marginTop": "24px", "padding": "0"
                }
            )
        ],
        id="pdf-report-section",
        style={"background": "#fff", "color": "#222", "padding": "18px"}
    )


from dash import clientside_callback, Output, Input


clientside_callback(
    """
    function(trigger) {
        if(trigger){
            setTimeout(() => window.print(), 250);
        }
        return false;
    }
    """,
    Output("trigger-print-store", "data"),
    Input("trigger-print-store", "data"),
    prevent_initial_call=True
)

from dash import callback_context

@app.callback(
    Output('uploaded-image-store', 'data'),
    Output('file-dropdown', 'value', allow_duplicate=True),  # <-- Ajoute cette ligne !
    Input('upload-image', 'contents'),
    Input('file-dropdown', 'value'),
    prevent_initial_call=True
)
def set_uploaded_image(contents, dropdown_value):
    triggered = callback_context.triggered_id if hasattr(callback_context, 'triggered_id') else None
    if triggered == "upload-image" and contents is not None:
        # Vide le dropdown quand upload local, pour forcer le switch visuel
        return contents, None
    elif triggered == "file-dropdown":
        return None, dropdown_value
    return None, None


# ====================================================
# POINT D'ENTRÉE DE L'APPLICATION
# ====================================================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
