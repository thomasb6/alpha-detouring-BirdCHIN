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

# Configuration pour l'édition des formes sur le graphique
config_graph = {
    "modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
    "displaylogo": False,
}

# L'application charge automatiquement le fichier CSS présent dans le dossier assets.
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="BirdChin")
server = app.server

filenames = get_filenames()
classification_options = ["plaque", "atrophique", "incertaine"]

app.layout = dbc.Container([
    # Bloc de gauche : Instructions
    html.Div([
        html.H1([html.Span("Instructions d'utilisation")]),
        html.P("1. Sélectionnez un fichier à analyser dans le menu déroulant."),
        html.P("2. Utilisez l'outil de dessin pour détourer une tache sur l'image."),
        html.P("3. Vous pouvez réinitialiser le détourage."),
        html.P("4. Attribuez une classification à chaque zone en cliquant sur l'un des boutons proposés après le détourage."),
        html.P("5. Vous pouvez modifier une classification en sélectionnant une Zone existante à partir du menu déroulant."),
        html.P("6. Vous pouvez modifier la taille d'une tache en la sélectionnant sur le dessin."),
        html.P("7. Cliquez sur 'Exporter vers Excel' pour télécharger un tableur contenant un résumé des zones pour une image.")
    ], className='left-block'),

    # Bloc du milieu : Graphique
    html.Div([
        dcc.Graph(
            id='fig-image',
            config=config_graph,
            className="graph-figure"
        ),
        html.Div(id='output-area', className="output-area")
    ], className='middle-block'),

    # Bloc de droite : Contrôles et interactions
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
        dcc.Dropdown(
            id="zone-selector",
            options=[],
            placeholder="Sélectionnez une zone à reclassifier"
        ),
        html.Br(),
        dbc.Button("Réinitialiser les annotations", id="reset-button", color="danger", className="mb-2"),
        html.Br(),
        dbc.Button("Exporter vers Excel", id="export-button", color="primary", className="mb-2"),
        dcc.Download(id="download-dataframe-xlsx"),
        html.Br(),
        dbc.Button("Télécharger les annotations", id="download-json-button", color="primary", className="mb-2"),
        dcc.Download(id="download-json"),
        dcc.Upload(
            id='upload-annotations',
            children=html.Div([
                'Glissez-déposez ou ',
                html.A('sélectionnez un fichier annoté', className="upload-link")
            ]),
            className="upload-area",
            multiple=False
        ),
        dcc.Input(
            id="key-capture",
            type="text",
            className="key-capture",
            autoFocus=True
        ),
        dcc.Store(id="stored-shapes", data=[]),
        html.Div(id='output-text', className="output-text")
    ], className='right-block')
],
    fluid=True,
    className='dashboard-container',
    style={'display': 'flex', 'justify-content': 'space-between'}
)

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
                color='white',  # Couleur similaire à celle du lasso (vous pouvez la modifier)
                width=2,  # Épaisseur fine du trait
                dash='dash'  # Trait en pointillé, comme le lasso
            )
        ),
        hovermode=False  # Désactive le hover dans l'ensemble de la figure
    )
    return fig

# --- Les callbacks restent inchangés ---

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

    if stored_shapes is not None:
        for shape in stored_shapes:
            shape.setdefault("editable", True)
            shape.setdefault("layer", "above")
            shape.setdefault("xref", "x")
            shape.setdefault("yref", "y")
            shape.setdefault("line", {"width": 0.1})  # Définit une épaisseur de trait de 1 pixel
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
    Output("key-capture", "value"),
    Input("fig-image", "relayoutData"),
    Input("reset-button", "n_clicks"),
    Input({"type": "classify-button", "index": ALL}, "n_clicks"),
    Input("key-capture", "value"),
    Input("upload-annotations", "contents"),
    State("stored-shapes", "data"),
    State("zone-selector", "value"),
    prevent_initial_call=True
)
def update_shapes(relayout_data, reset_clicks, classify_clicks, key_value, upload_contents, stored_shapes, selected_zone):
    trigger = ctx.triggered_id

    if stored_shapes is None:
        stored_shapes = []

    if trigger == "upload-annotations" and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            annotations = json.loads(decoded.decode('utf-8'))
            stored_shapes = annotations
        except Exception as e:
            print(f"Erreur lors du chargement des annotations : {e}")
        areas = []
        for i, shape in enumerate(stored_shapes):
            path_str = shape.get("path", "")
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
            try:
                coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
                area = calculate_area(coords)
                lab = shape.get("customdata", "Tache")
                areas.append(f"Zone {i+1} : {area:.2f} pixels² ({lab})")
            except Exception as e:
                areas.append(f"Zone {i+1} : erreur ({e})")
        return stored_shapes, html.Ul([html.Li(a) for a in areas]), key_value

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
            for key, value in relayout_data.items():
                if key.startswith("shapes["):
                    match = re.match(r"shapes\[(\d+)\]\.(\w+)", key)
                    if match:
                        index = int(match.group(1))
                        prop = match.group(2)
                        if index < len(stored_shapes):
                            stored_shapes[index][prop] = value

    areas = []
    for i, shape in enumerate(stored_shapes):
        path_str = shape.get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j+1])) for j in range(0, len(matches), 2)]
            area = calculate_area(coords)
            lab = shape.get("customdata", "Tache")
            areas.append(f"Zone {i+1} : {area:.2f} pixels² ({lab})")
        except Exception as e:
            areas.append(f"Zone {i+1} : erreur ({e})")
    return stored_shapes, html.Ul([html.Li(a) for a in areas]), key_value

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


if __name__ == '__main__':
    app.run(debug=False)
