# ====================================================
# IMPORTS ET CONFIGURATIONS INITIALS
# ====================================================

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

# ====================================================
# CONFIGURATION DE L'ACCÈS AU RÉPERTOIRE GITHUB
# ====================================================
REPO_OWNER = "thomasb6"
REPO_NAME = "alpha-detouring-BirdCHIN"
FOLDER_PATH = "cropped"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FOLDER_PATH}"
GITHUB_TOKEN = "ghp_nwTO1ndY???????????rsxh9HxEJKi2QiZNDWGCSX?3?z?U?g?NP"
GITHUB_TOKEN = GITHUB_TOKEN.replace("?", "")

# ====================================================
# FONCTIONS AUXILIAIRES POUR LA GESTION DES IMAGES ET DES COORDONNÉES
# ====================================================
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

config_graph = {
    "modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
    "displaylogo": False,
}

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


# ====================================================
# CONFIGURATION DE L'APPLICATION ET DU THÈME
# ====================================================
external_stylesheets = [
    dbc.themes.FLATLY,
    "https://use.fontawesome.com/releases/v5.15.3/css/all.css"
]

app = Dash(__name__, external_stylesheets=external_stylesheets, title="BirdChin")
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
app.layout = html.Div([
    dbc.Container([
        html.Div([
            html.Div([
                html.Img(src=app.get_asset_url('logo.png'),
                         style={
                             'height': '80px',
                             'verticalAlign': 'middle',
                             'marginRight': '10px'
                         }),
                html.Span("BirdChin", style={"fontSize": "37px", "verticalAlign": "middle"}),
            ], className="logo-container"),
            html.H2("Instructions d'utilisation"),
            html.P("1. Choisissez une image depuis le menu déroulant."),
            html.P("2. Tracez le contour d'une lésion sur l'image."),
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
            html.P("Choix de l'image :"),
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': f, 'value': f} for f in filenames],
                placeholder='Sélectionnez un fichier à analyser'
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
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Label("Rotation globale (°) :"),
            dcc.Slider(
                id='rotation-slider',
                min=-30, max=30, step=0.5, value=0,
                tooltip={"placement": "bottom", "always_visible": True}
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
            dcc.Store(id="stored-shapes", data=[]),
            html.Div(id='output-text', className="output-text")
        ], className='right-block')
    ],
        fluid=True,
        className='dashboard-container',
        style={'display': 'flex', 'justify-content': 'space-between'}
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
    )
])

# ====================================================
# FONCTION POUR GÉNÉRER LA FIGURE À PARTIR D'UNE IMAGE
# ====================================================
def generate_figure(image, file_val=None):
    fig = px.imshow(image)
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    fig.update_layout(
        dragmode="drawclosedpath",
        uirevision=file_val or str(random.random()),  # valeur différente pour chaque image
        paper_bgcolor='black',
        plot_bgcolor='black',
        width=700,
        height=700,
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=[],
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
# CALLBACK 1 : MISE À JOUR DE LA FIGURE D'AFFICHAGE
# ====================================================
@app.callback(
    Output("fig-image", "figure"),
    Input("file-dropdown", "value"),
    Input("reset-button", "n_clicks"),
    Input("stored-shapes", "data"),
    Input("show-zone-numbers", "checked"),
    Input("dashed-contour", "checked"),
    Input("zoom-slider", "value"),
    Input("rotation-slider", "value"),
    State("fig-image", "figure")
)
def update_figure(file_val, reset_clicks, stored_shapes, show_zone_numbers, dashed_contour, zoom, rotation, current_fig):
    if file_val:
        try:
            image_url = get_image_url(file_val)
            response = requests.get(image_url)
            image = Image.open(io_buffer.BytesIO(response.content))
            fig = generate_figure(image, file_val=file_val)
            width, height = image.size
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Erreur lors du chargement : {e}")
            width, height = 700, 700
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

        # Annotations pour les numéros
        def centroid(coords):
            if not coords:
                return 0, 0
            avg_x = sum(x for x, y in coords) / len(coords)
            avg_y = sum(y for x, y in coords) / len(coords)
            return avg_x, avg_y

        annotations = []
        for i, shape in enumerate(stored_shapes):
            # Applique la même transformation aux coords pour les annotations
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
    Input("add-nerf-optique-button", "n_clicks"),
    Input("fig-image", "relayoutData"),
    Input("reset-button", "n_clicks"),
    Input({"type": "classify-button", "index": ALL}, "n_clicks"),
    Input("upload-annotations", "contents"),
    Input("file-dropdown", "value"),
    State("stored-shapes", "data"),
    State("zone-selector", "value"),
    prevent_initial_call=True
)
def update_shapes_combined(
    add_nerf_clicks,      # 1
    relayout_data,        # 2
    reset_clicks,         # 3
    classify_clicks,      # 4
    upload_contents,      # 5
    file_val,             # 6
    stored_shapes,        # 7
    selected_zone         # 8
):
    trigger = ctx.triggered_id
    if stored_shapes is None:
        stored_shapes = []
    new_upload = dash.no_update

    # ----------- OUVERTURE IMAGE -----------
    if trigger == "file-dropdown" and file_val:
        if not stored_shapes or len(stored_shapes) == 0:
            # Première ouverture, on ajoute le nerf optique
            try:
                image_url = get_image_url(file_val)
                response = requests.get(image_url)
                img = Image.open(io_buffer.BytesIO(response.content))
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
            stored_shapes = [cercle_nerf]
        summary = générer_resume(stored_shapes)
        return stored_shapes, summary, new_upload

    # ----------- RESET -----------
    elif trigger == "reset-button":
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
        return [], "Annotations réinitialisées.", new_upload

    # ----------- IMPORT JSON -----------
    elif trigger == "upload-annotations" and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            new_annotations = json.loads(decoded.decode('utf-8'))
        except Exception:
            new_annotations = []
        # On REMPLACE toutes les anciennes annotations
        stored_shapes = []
        for i, shape in enumerate(new_annotations):
            if "customid" not in shape:
                shape["customid"] = i + 1
            stored_shapes.append(shape)
        summary = générer_resume(stored_shapes)
        return stored_shapes, summary, new_upload

    # ----------- AJOUT NERF OPTIQUE À LA DEMANDE -----------
    elif trigger == "add-nerf-optique-button":
        # On ne doit ajouter le nerf optique que s'il n'existe pas déjà
        if file_val:
            try:
                image_url = get_image_url(file_val)
                response = requests.get(image_url)
                img = Image.open(io_buffer.BytesIO(response.content))
                width, height = img.size
            except Exception:
                width, height = 700, 700
        else:
            width, height = 700, 700
        cx, cy = width / 2, height / 2
        r = 50  # rayon par défaut
        # Vérifie si une zone "nerf optique" existe déjà
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
            stored_shapes = [cercle_nerf] + stored_shapes
        summary = générer_resume(stored_shapes)
        return stored_shapes, summary, new_upload

    # ----------- CLASSIFICATION -----------
    elif isinstance(trigger, dict) and trigger.get("type") == "classify-button":
        label = trigger["index"]
        if selected_zone is not None and selected_zone < len(stored_shapes):
            stored_shapes[selected_zone]["customdata"] = label
        elif stored_shapes:
            stored_shapes[-1]["customdata"] = label

    # ----------- DESSIN / MODIF FORME -----------
    elif relayout_data:
        if "shapes" in relayout_data:
            new_shapes = relayout_data["shapes"]
            updated_shapes = []
            for i, new_shape in enumerate(new_shapes):
                valid = {k: v for k, v in new_shape.items() if k not in ["customdata", "customid"]}
                valid["customdata"] = stored_shapes[i].get("customdata", "Tache") if i < len(stored_shapes) else "Tache"
                if "customid" not in valid:
                    valid["customid"] = len(stored_shapes) + 1
                updated_shapes.append(valid)
            stored_shapes = updated_shapes
        else:
            import re
            for key, val in relayout_data.items():
                m = re.match(r"shapes\[(\d+)\]\.(\w+)", key)
                if m:
                    idx, prop = int(m.group(1)), m.group(2)
                    if idx < len(stored_shapes):
                        stored_shapes[idx][prop] = val

    # ----------- PAR DÉFAUT : REFRESH -----------
    summary = générer_resume(stored_shapes)
    return stored_shapes, summary, new_upload

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
        image_url = get_image_url(file_val)
        response = requests.get(image_url)
        image = Image.open(io_buffer.BytesIO(response.content))
        # Redimensionnement pour conserver la cohérence avec le rendu (ici 700x700)
        #image = image.resize((700, 700))
        width, height = image.size
        nerf_optique_centroid = (width / 2, height / 2)
    except Exception as e:
        # En cas d'erreur, on définit une valeur par défaut
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

# ====================================================
# POINT D'ENTRÉE DE L'APPLICATION
# ====================================================
if __name__ == '__main__':
    app.run(debug=False)
