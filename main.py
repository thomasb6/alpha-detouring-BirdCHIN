from dash import Dash, html, dcc, Input, Output, State, ctx, ALL
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
import json
import base64

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
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(coords)-1)))

# Figure de démarrage (affichée tant qu'aucun fichier n'est sélectionné)
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

# Configuration pour l'édition des formes sur le graphique
config_graph = {
    "modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
    "displaylogo": False,
}

# Ajout du thème FLATLY et de Font Awesome pour les icônes
external_stylesheets = [
    dbc.themes.FLATLY,
    "https://use.fontawesome.com/releases/v5.15.3/css/all.css"
]

# L'application charge automatiquement le fichier CSS présent dans le dossier assets.
app = Dash(__name__, external_stylesheets=external_stylesheets, title="BirdChin")
server = app.server

filenames = get_filenames()
classification_options = ["plaque", "atrophique", "incertaine"]
# Dictionnaire de correspondance entre le libellé et la touche souhaitée
shortcut_keys = {"plaque": "p", "atrophique": "a", "incertaine": "i"}

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



# Layout principal
app.layout = html.Div([
    dbc.Container([
        # Bloc de gauche : Instructions + entête avec logo et titre
        html.Div([
            html.Div([
                html.Img(src=app.get_asset_url('logo.png'),
                         style={
                             'height': '80px',  # Logo agrandi
                             'verticalAlign': 'middle',
                             'marginRight': '10px'
                         }),
                html.Span("BirdChin", style={"fontSize": "37px", "verticalAlign": "middle"}),
            ], className="logo-container"),  # Conteneur pour le logo et le titre
            html.H2([html.Span("Instructions d'utilisation")]),
            html.P("1. Choisissez une image depuis le menu déroulant."),
            html.P("2. Tracez le contour du nerf optique sur l'image."),
            html.P("3. Tracez le contour d'une lésion sur l'image."),
            html.P("4. Classez la zone en cliquant sur le type approprié."),
            html.H3("Vous pouvez supprimer une zone en la sélectionnant."),
            html.H3("Vous pouvez modifier une classification via le menu déroulant."),
            html.P("5. Exportez les résultats vers Excel pour obtenir un résumé."),
            html.P("6. Téléchargez les zones annotées."),
            html.H3("Vous pouvez importer un fichier avec les zones annotées."),
        ], className='left-block'),

        # Bloc du milieu : Graphique
        html.Div([
            dcc.Graph(
                id='fig-image',
                config=config_graph,
                style={'width': '100%', 'height': 'auto'},
                className="graph-figure"
            ),
            html.Div(id='output-area', className="output-area")
        ], className='middle-block'),

        # Bloc de droite : Contrôles et interactions
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
            html.P("Importer :"),
            dcc.Upload(
                id='upload-annotations',
                children=html.Div([
                    html.I(className="fas fa-upload", style={"margin-right": "5px"}),
                    "Glissez-déposez ou sélectionnez un fichier annoté"
                ]),
                className="upload-area",
                style={"width": "100%"},
                multiple=False
            ),
            dcc.Store(id="stored-shapes", data=[]),
            html.Div(id='output-text', className="output-text")
        ], className='right-block')
    ],
    fluid=True,
    className='dashboard-container',
    style={'display': 'flex', 'justify-content': 'space-between'}
    ),

    # Footer
    html.Footer(
        html.Div([
            "© 2025 – Réalisé par ",
            html.A(
                "Thomas Foulonneau",
                href="https://www.linkedin.com/in/thomas-foulonneau?originalSubdomain=fr",
                target="_blank",
                style={
                    "color": "#ffffff",
                    "textDecoration": "underline"
                }
            ),
            " – Interne à l'Ophtalmopole de Paris"
        ]),
        className="footer"
    )

])

def generate_figure(image):
    fig = px.imshow(image)
    # Désactivation des infos de survol pour chaque trace
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    # Désactivation globale du mode hover
    fig.update_layout(
        dragmode="drawclosedpath",
        uirevision="constant",
        paper_bgcolor='black',
        plot_bgcolor='black',
        width=image.width,
        height=image.height,
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=[],
        newshape=dict(
            line=dict(
                color='white',  # Couleur similaire à celle du lasso (modifiable)
                width=2,         # Épaisseur fine du trait
                dash='dash'      # Trait en pointillé
            )
        ),
        hovermode=False
    )
    return fig

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
                image = image.resize((700, 700))
                fig = generate_figure(image)
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {str(e)}")
    else:
        fig = current_fig if current_fig is not None else scatter_fig

    if stored_shapes is not None:
        for shape in stored_shapes:
            shape.setdefault("editable", True)
            shape.setdefault("layer", "above")
            shape.setdefault("xref", "x")
            shape.setdefault("yref", "y")
            shape.setdefault("line", {"width": 0.1})
        fig["layout"]["shapes"] = stored_shapes

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
                coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
            except Exception:
                continue
            cx, cy = centroid(coords)
            annotations.append(dict(
                x=cx,
                y=cy,
                text=str(i+1),
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
                font=dict(color="white", size=12)
            ))
        fig["layout"]["annotations"] = annotations
    return fig


@app.callback(
    Output("stored-shapes", "data"),
    Output("output-area", "children"),
    Input("fig-image", "relayoutData"),
    Input("reset-button", "n_clicks"),
    Input({"type": "classify-button", "index": ALL}, "n_clicks"),
    Input("upload-annotations", "contents"),
    State("stored-shapes", "data"),
    State("zone-selector", "value"),
    prevent_initial_call=True
)
def update_shapes(relayout_data, reset_clicks, classify_clicks, upload_contents, stored_shapes, selected_zone):
    trigger = ctx.triggered_id

    if stored_shapes is None:
        stored_shapes = []

    # Si l'utilisateur charge un fichier JSON
    if trigger == "upload-annotations" and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            new_annotations = json.loads(decoded.decode('utf-8'))
        except Exception as e:
            new_annotations = []
            print(f"Erreur lors du chargement des annotations : {e}")
        for shape in new_annotations:
            if "customid" not in shape:
                shape["customid"] = len(stored_shapes) + 1
        stored_shapes.extend(new_annotations)
        summary = générer_resume(stored_shapes)
        return stored_shapes, summary

    # Si l'utilisateur réinitialise les annotations
    elif trigger == "reset-button":
        return [], "Annotations réinitialisées."

    # Si le trigger est un clic sur un bouton de classification
    elif isinstance(trigger, dict) and trigger.get("type") == "classify-button":
        label = trigger["index"]
        if selected_zone is not None and selected_zone < len(stored_shapes):
            stored_shapes[selected_zone]["customdata"] = label
        elif stored_shapes:
            stored_shapes[-1]["customdata"] = label

    # Si l'utilisateur modifie une annotation via l'outil de dessin (relayoutData)
    elif relayout_data:
        if "shapes" in relayout_data:
            new_shapes = relayout_data["shapes"]
            updated_shapes = []
            for i, new_shape in enumerate(new_shapes):
                if i < len(stored_shapes):
                    new_shape["customdata"] = stored_shapes[i].get("customdata", "Tache")
                else:
                    new_shape["customdata"] = "Tache"
                if "customid" not in new_shape:
                    new_shape["customid"] = len(stored_shapes) + 1
                updated_shapes.append(new_shape)
            stored_shapes = updated_shapes
        else:
            # Mise à jour des propriétés d'une annotation existante
            for key, value in relayout_data.items():
                if key.startswith("shapes["):
                    match = re.match(r"shapes\[(\d+)\]\.(\w+)", key)
                    if match:
                        index = int(match.group(1))
                        prop = match.group(2)
                        if index < len(stored_shapes):
                            stored_shapes[index][prop] = value

    summary = générer_resume(stored_shapes)
    return stored_shapes, summary

def générer_resume(shapes):
    areas = []
    for i, shape in enumerate(shapes):
        path_str = shape.get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
            area = calculate_area(coords)
            lab = shape.get("customdata", "Tache")
            areas.append(f"Zone {i+1} : {area:.2f} pixels² ({lab})")
        except Exception as e:
            areas.append(f"Zone {i+1} : erreur ({e})")
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

@app.callback(
    Output("zone-selector", "options"),
    Input("stored-shapes", "data")
)
def update_zone_selector_options(stored_shapes):
    if stored_shapes is None:
        return []
    return [{"label": f"Zone {i+1}", "value": i} for i in range(len(stored_shapes))]

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

    import numpy as np

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
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        return centroid, major_axis, minor_axis, angle

    def compute_angle_diff(centroid, angle, reference_centroid):
        dx = reference_centroid[0] - centroid[0]
        dy = reference_centroid[1] - centroid[1]
        ref_angle = np.degrees(np.arctan2(dy, dx))
        angle_diff = angle - ref_angle
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff

    rows = []
    nerf_optique_centroid = None
    if len(stored_shapes) > 0:
        path_str = stored_shapes[0].get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
            nerf_optique_centroid = calc_centroid(coords)
        except Exception:
            nerf_optique_centroid = None

    for i, shape in enumerate(stored_shapes):
        path_str = shape.get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
            area = calculate_area(coords)
        except Exception:
            area = None
            coords = []
        cx, cy = calc_centroid(coords) if coords else (None, None)
        classification = shape.get("customdata", "Tache")
        try:
            if len(coords) >= 2:
                centroid, major_axis, minor_axis, angle = compute_ellipse_params(coords)
            else:
                major_axis = None
                minor_axis = None
                angle = None
        except Exception:
            major_axis = None
            minor_axis = None
            angle = None

        if nerf_optique_centroid is not None and cx is not None and angle is not None:
            angle_diff = compute_angle_diff((cx, cy), angle, nerf_optique_centroid)
        else:
            angle_diff = None

        rows.append({
            "Zone": i+1,
            "Aire (pixels²)": area,
            "Centroid X": cx,
            "Centroid Y": cy,
            "Classification": classification,
            "Grand Axe (pixels)": major_axis,
            "Petit Axe (pixels)": minor_axis,
            "Angle (degrés) par rapport Nerf Optique": angle_diff
        })
    df = pd.DataFrame(rows)
    filename = f"{file_val.split('.')[0]}.xlsx" if file_val else "export.xlsx"
    def to_excel(bytes_io):
        with pd.ExcelWriter(bytes_io, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Zones")
    return dcc.send_bytes(to_excel, filename)

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
    Output('upload-div', 'children'),
    Input('upload-annotations', 'contents'),
    prevent_initial_call=True
)
def reset_upload(contents):
    if contents:
        return [
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
    return dash.no_update

if __name__ == '__main__':
    app.run(debug=False)
