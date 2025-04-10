# ====================================================
# IMPORTS ET CONFIGURATIONS INITIALS
# ====================================================

# Importation des composants Dash pour construire l'application web
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL
# Composants Bootstrap pour styliser l'interface utilisateur
import dash_bootstrap_components as dbc
# Bibliothèques Plotly pour créer des graphiques interactifs
import plotly.graph_objects as go
import plotly.express as px
# Numpy pour les opérations mathématiques et la génération de valeurs aléatoires
import numpy as np
import random
# Requests pour interagir avec l'API GitHub afin de récupérer les images
import requests
# Module io renommé pour utiliser des tampons de données binaires
import io as io_buffer
# Pillow pour le traitement d'images
from PIL import Image
# Expression régulière afin d'extraire les coordonnées depuis les chaînes de caractères
import re
# Importation de Dash (redondant avec la première importation, mais souvent utilisé pour certaines méthodes)
import dash
# Pandas pour la création/export d'Excel contenant les données d'annotations
import pandas as pd
# JSON et base64 pour le traitement des fichiers d'annotations (import/export)
import json
import base64

# ====================================================
# CONFIGURATION DE L'ACCÈS AU RÉPERTOIRE GITHUB
# ====================================================
# Ces variables configurent l'accès à un répertoire GitHub contenant les fichiers images (.jpg)
REPO_OWNER = "thomasb6"
REPO_NAME = "alpha-detouring-BirdCHIN"
FOLDER_PATH = "Optos_1004"
# API GitHub pour accéder au contenu du dossier
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FOLDER_PATH}"
# Clé d'accès personnelle à l'API GitHub (attention à la sécurité lors de son usage en production)
GITHUB_TOKEN = "ghp_nwTO1ndYrsxh9HxEJKi2QiZNDWGCSX?3?z?U?g?NP"
# Suppression des caractères "?" indésirables dans le token
GITHUB_TOKEN = GITHUB_TOKEN.replace("?", "")


# ====================================================
# FONCTIONS AUXILIAIRES POUR LA GESTION DES IMAGES ET DES COORDONNÉES
# ====================================================

def get_filenames():
    """
    Récupère la liste des noms de fichiers situés dans le répertoire GitHub défini.
    Utilise l'API GitHub pour obtenir le contenu du dossier et filtre uniquement les fichiers.
    """
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(GITHUB_API_URL, headers=headers)
    if response.status_code == 200:
        # Retourne uniquement le nom des fichiers dont le type est "file"
        return [file["name"] for file in response.json() if file["type"] == "file"]
    return []


def get_image_url(filename):
    """
    Construit l'URL pour accéder au contenu brut (raw) de l'image depuis GitHub.
    :param filename: nom du fichier image.
    """
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{FOLDER_PATH}/{filename}"


def calculate_area(coords):
    """
    Calcule l'aire d'un polygone défini par une liste de coordonnées
    en utilisant la formule de Gauss (également appelée formule de shoelace).

    :param coords: liste de tuples (x, y) définissant le contour du polygone.
    :return: aire calculée en unités de pixels².
    """
    if len(coords) < 3:
        return 0
    x, y = zip(*coords)
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(coords) - 1)))


# ====================================================
# CONFIGURATION DE LA FIGURE INITIALE (AFFICHÉE EN L'ATTENTE D'UNE IMAGE)
# ====================================================
# Création d'une figure de départ avec un nuage de points générés aléatoirement
scatter_fig = go.Figure(
    go.Scattergl(
        x=np.random.randn(1000),
        y=np.random.randn(1000),
        mode='markers',
        marker=dict(
            # Couleurs alternées pour fournir un rendu visuel stylisé
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

# Configuration pour l'outil de dessin dans le graphique
config_graph = {
    "modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
    "displaylogo": False,
}

# ====================================================
# CONFIGURATION DE L'APPLICATION ET DU THÈME
# ====================================================
# Ajout d'un thème Bootstrap (FLATLY) et d'icônes Font Awesome pour améliorer l'interface
external_stylesheets = [
    dbc.themes.FLATLY,
    "https://use.fontawesome.com/releases/v5.15.3/css/all.css"
]

# L'application charge automatiquement les fichiers CSS présents dans le dossier "assets"
app = Dash(__name__, external_stylesheets=external_stylesheets, title="BirdChin")
server = app.server  # Pour le déploiement sur un serveur compatible WSGI

# Récupération de la liste des fichiers images depuis le répertoire GitHub
filenames = get_filenames()

# Options de classification pour les annotations d'image
classification_options = ["grande", "atrophique", "pigmentée", "incertaine"]
# Dictionnaire mappant chaque libellé à une touche de raccourci
shortcut_keys = {"grande": "g", "atrophique": "a", "pigmentée" : "m", "incertaine": "i"}

# Création des boutons de classification à afficher
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
# DÉFINITION DU LAYOUT PRINCIPAL DE L'APPLICATION
# ====================================================
app.layout = html.Div([
    dbc.Container([
        # ------------------------
        # Bloc de gauche : Instructions et entête avec logo et titre
        # ------------------------
        html.Div([
            html.Div([
                html.Img(src=app.get_asset_url('logo.png'),
                         style={
                             'height': '80px',  # Agrandissement du logo
                             'verticalAlign': 'middle',
                             'marginRight': '10px'
                         }),
                html.Span("BirdChin", style={"fontSize": "37px", "verticalAlign": "middle"}),
            ], className="logo-container"),  # Conteneur pour regrouper logo et titre
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

        # ------------------------
        # Bloc du milieu : Graphique pour l'affichage des images et annotations
        # ------------------------
        html.Div([
            dcc.Graph(
                id='fig-image',
                config=config_graph,
                style={'width': '100%', 'height': 'auto'},
                className="graph-figure"
            ),
            # Zone d'affichage du résumé des annotations
            html.Div(id='output-area', className="output-area")
        ], className='middle-block'),

        # ------------------------
        # Bloc de droite : Contrôles et interactions
        # ------------------------
        html.Div([
            html.P("Choix de l'image :"),
            # Dropdown pour sélectionner une image parmi celles récupérées de GitHub
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': f, 'value': f} for f in filenames],
                placeholder='Sélectionnez un fichier à analyser'
            ),
            html.P("Classification :"),
            # Groupe de boutons pour classifier la zone dessinée
            dbc.ButtonGroup(
                classification_buttons,
                vertical=False,
                className="mb-2",
                style={"width": "100%", "display": "flex"}
            ),
            # Dropdown pour sélectionner une zone à reclassifier
            dcc.Dropdown(
                id="zone-selector",
                options=[],
                placeholder="Sélectionnez une zone à reclassifier"
            ),
            html.P("Réinitialiser :"),
            # Bouton pour réinitialiser (effacer) les annotations
            dbc.Button([
                html.I(className="fas fa-undo", style={"margin-right": "5px"}),
                "Réinitialiser les zones annotées"
            ], id="reset-button", color="danger", className="mb-2"),
            html.P("Exporter :"),
            # Bouton pour exporter les résultats sous format Excel
            dbc.Button([
                html.I(className="fas fa-download", style={"margin-right": "5px"}),
                "Exporter les résultats dans un tableur"
            ], id="export-button", color="primary", className="mb-2"),
            dcc.Download(id="download-dataframe-xlsx"),
            # Bouton pour exporter les annotations au format JSON
            dbc.Button([
                html.I(className="fas fa-file-export", style={"margin-right": "5px"}),
                "Exporter les annotations"
            ], id="download-json-button", color="primary", className="mb-2"),
            dcc.Download(id="download-json"),
            html.P("Importer :"),
            # Zone d'upload pour importer des annotations sous format JSON
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
            # Stockage local des données d'annotations (shapes)
            dcc.Store(id="stored-shapes", data=[]),
            html.Div(id='output-text', className="output-text")
        ], className='right-block')
    ],
        fluid=True,
        className='dashboard-container',
        style={'display': 'flex', 'justify-content': 'space-between'}
    ),

    # ------------------------
    # Footer de l'application
    # ------------------------
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


# ====================================================
# FONCTION DE GÉNÉRATION DE LA FIGURE D'IMAGE
# ====================================================

def generate_figure(image):
    """
    Génère une figure Plotly à partir d'une image.
    Utilise plotly.express pour afficher l'image, désactive les informations de survol
    et configure le mode de dessin pour ajouter des formes (annotations).

    :param image: image au format PIL.Image
    :return: objet figure configuré pour l'annotation
    """
    # Affichage de l'image avec plotly.express
    fig = px.imshow(image)
    # Désactivation des infos de survol sur toutes les traces
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    # Configuration de la mise en page de la figure
    fig.update_layout(
        dragmode="drawclosedpath",  # Mode de dessin pour tracer des formes fermées
        uirevision="constant",  # Permet de conserver l'état de la figure lors des mises à jour
        paper_bgcolor='black',
        plot_bgcolor='black',
        width=image.width,
        height=image.height,
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=[],  # Initialisation sans forme, les formes seront ajoutées ultérieurement
        newshape=dict(
            line=dict(
                color='white',  # Couleur de la ligne de dessin (modifiable)
                width=2,  # Épaisseur de la ligne en mode dessin
                dash='dash'  # Style de ligne en pointillés
            )
        ),
        hovermode=False
    )
    return fig


# ====================================================
# CALLBACK 1 : MISE À JOUR DE LA FIGURE D'AFFICHAGE
# ====================================================
@app.callback(
    Output("fig-image", "figure"),
    Input("file-dropdown", "value"),
    Input("reset-button", "n_clicks"),
    Input("stored-shapes", "data"),
    State("fig-image", "figure")
)
def update_figure(file_val, reset_clicks, stored_shapes, current_fig):
    """
    Met à jour la figure affichée.
      - Si un nouveau fichier est sélectionné ou que l'utilisateur réinitialise les annotations,
        la fonction charge l'image correspondante et génère une nouvelle figure.
      - Sinon, elle affiche la figure courante en conservant éventuellement les annotations (shapes).
    """
    trigger = ctx.triggered_id  # Identifie l'élément qui a déclenché le callback
    if trigger in ["file-dropdown", "reset-button"]:
        if not file_val:
            # Si aucun fichier n'est sélectionné, afficher la figure de démarrage (scatter_fig)
            fig = scatter_fig
        else:
            try:
                # Récupération de l'image en utilisant l'URL construite dynamiquement
                image_url = get_image_url(file_val)
                response = requests.get(image_url)
                image = Image.open(io_buffer.BytesIO(response.content))
                image = image.resize((700, 700))
                fig = generate_figure(image)
            except Exception as e:
                # En cas d'erreur (problème d'accès ou de traitement de l'image)
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {str(e)}")
    else:
        # Si aucune nouvelle action n'est déclenchée, la figure reste inchangée
        fig = current_fig if current_fig is not None else scatter_fig

    # Si des annotations ("stored_shapes") existent, les ajouter à la figure
    if stored_shapes is not None:
        for shape in stored_shapes:
            # S'assurer que chaque forme est éditable et bien positionnée
            shape.setdefault("editable", True)
            shape.setdefault("layer", "above")
            shape.setdefault("xref", "x")
            shape.setdefault("yref", "y")
            shape.setdefault("line", {"width": 0.1})
        fig["layout"]["shapes"] = stored_shapes

        def centroid(coords):
            """
            Calcule le centroïde d'une liste de coordonnées.
            """
            if not coords:
                return 0, 0
            avg_x = sum(x for x, y in coords) / len(coords)
            avg_y = sum(y for x, y in coords) / len(coords)
            return avg_x, avg_y

        annotations = []
        # Ajout d'une annotation textuelle pour chaque forme (numérotation des zones)
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


# ====================================================
# CALLBACK 2 : MISE À JOUR ET GESTION DES ANNOTATIONS (SHAPES)
# ====================================================
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
    """
    Met à jour la liste des formes (annotations) selon plusieurs actions :
      - Chargement d'un fichier JSON contenant des annotations.
      - Réinitialisation (effacement) des annotations.
      - Classification d'une zone via des boutons dédiés.
      - Modification des annotations via l'outil de dessin.

    Retourne les annotations mises à jour et un résumé visuel sous forme de Card.
    """
    trigger = ctx.triggered_id  # Identification de l'action déclenchante

    if stored_shapes is None:
        stored_shapes = []

    # Si l'utilisateur charge un fichier JSON contenant des annotations
    if trigger == "upload-annotations" and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            new_annotations = json.loads(decoded.decode('utf-8'))
        except Exception as e:
            new_annotations = []
            print(f"Erreur lors du chargement des annotations : {e}")
        # Assure que chaque annotation possède un identifiant personnalisé
        for shape in new_annotations:
            if "customid" not in shape:
                shape["customid"] = len(stored_shapes) + 1
        stored_shapes.extend(new_annotations)
        summary = générer_resume(stored_shapes)
        return stored_shapes, summary

    # Réinitialisation des annotations sur clic du bouton "reset"
    elif trigger == "reset-button":
        return [], "Annotations réinitialisées."

    # Traitement d'un clic sur un bouton de classification
    elif isinstance(trigger, dict) and trigger.get("type") == "classify-button":
        label = trigger["index"]
        # Si une zone est sélectionnée dans le dropdown, classification de cette zone
        if selected_zone is not None and selected_zone < len(stored_shapes):
            stored_shapes[selected_zone]["customdata"] = label
        # Sinon, appliquer la classification à la dernière zone dessinée
        elif stored_shapes:
            stored_shapes[-1]["customdata"] = label

    # Mise à jour via l'outil de dessin : modification ou création de formes
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
            # Mise à jour des propriétés d'une annotation déjà existante
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
    """
    Génère un résumé sous forme de Card contenant :
      - Le numéro de la zone annotée.
      - L'aire calculée pour chaque zone.
      - La classification (s'il y a lieu).

    Le résumé est construit sous forme de liste HTML intégrée dans une Card Bootstrap.
    """
    areas = []
    for i, shape in enumerate(shapes):
        path_str = shape.get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
            area = calculate_area(coords)
            lab = shape.get("customdata", "Tache")
            areas.append(f"Zone {i + 1} : {area:.2f} pixels² ({lab})")
        except Exception as e:
            areas.append(f"Zone {i + 1} : erreur ({e})")
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
    """
    Met à jour les options du dropdown 'zone-selector' en fonction du nombre d'annotations existantes.
    Chaque option est une zone numérotée.
    """
    if stored_shapes is None:
        return []
    return [{"label": f"Zone {i + 1}", "value": i} for i in range(len(stored_shapes))]


# ====================================================
# CALLBACK 4 : EXPORT DES DONNÉES D'ANNOTATIONS VERS UN FICHIER EXCEL
# ====================================================
@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("export-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    prevent_initial_call=True
)
def export_to_excel(n_clicks, stored_shapes, file_val):
    """
    Exporte les annotations (shapes) vers un fichier Excel.
    Calcule pour chaque zone :
      - L'aire.
      - Le centroïde.
      - Les paramètres d'une ellipse approchante (axes majeur, mineur et angle).
      - L'angle relatif par rapport au nerf optique.

    Le résultat est stocké dans un DataFrame Pandas, ensuite écrit dans un fichier Excel.
    """
    if not n_clicks or not stored_shapes:
        return dash.no_update

    import numpy as np

    def calc_centroid(coords):
        arr = np.array(coords)
        if len(arr) == 0:
            return None, None
        return np.mean(arr, axis=0)

    def compute_ellipse_params(coords):
        """
        Calcule les paramètres pour une ellipse approchante basée sur l'ensemble de coordonnées.
        Retourne le centroïde, la longueur de l'axe majeur, de l'axe mineur et l'angle (en degrés).
        """
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
        """
        Calcule la différence d'angle entre l'orientation d'une zone annotée
        et la direction reliant le centroïde de la zone au nerf optique.
        """
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
    # Considère la première annotation comme le nerf optique
    if len(stored_shapes) > 0:
        path_str = stored_shapes[0].get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
            nerf_optique_centroid = calc_centroid(coords)
        except Exception:
            nerf_optique_centroid = None

    # Pour chaque annotation, calculs des métriques et stockage dans une liste de dictionnaires
    for i, shape in enumerate(stored_shapes):
        path_str = shape.get("path", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        try:
            coords = [(float(matches[j]), float(matches[j + 1])) for j in range(0, len(matches), 2)]
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
            "Zone": i + 1,
            "Aire (pixels²)": area,
            "Centroid X": cx,
            "Centroid Y": cy,
            "Classification": classification,
            "Grand Axe (pixels)": major_axis,
            "Petit Axe (pixels)": minor_axis,
            "Angle (degrés) par rapport Nerf Optique": angle_diff
        })
    # Création d'un DataFrame avec toutes les mesures pour export
    df = pd.DataFrame(rows)
    # Détermine le nom du fichier exporté en fonction du fichier sélectionné
    filename = f"{file_val.split('.')[0]}.xlsx" if file_val else "export.xlsx"

    def to_excel(bytes_io):
        # Utilisation de la librairie openpyxl pour écrire le DataFrame dans un fichier Excel
        with pd.ExcelWriter(bytes_io, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Zones")

    # Retourne le fichier Excel sous forme de bytes pour le téléchargement
    return dcc.send_bytes(to_excel, filename)


# ====================================================
# CALLBACK 5 : EXPORT DES ANNOTATIONS SOUS FORMAT JSON
# ====================================================
@app.callback(
    Output("download-json", "data"),
    Input("download-json-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    prevent_initial_call=True
)
def download_annotations(n_clicks, stored_shapes, file_val):
    """
    Permet de télécharger l'ensemble des annotations sous forme d'un fichier JSON.
    Le nom du fichier est généré dynamiquement en fonction du fichier image sélectionné.
    """
    if not stored_shapes:
        return dash.no_update
    content = json.dumps(stored_shapes)
    filename = f"{file_val.split('.')[0]}.json" if file_val else "annotations.json"
    return dcc.send_string(content, filename)


# ====================================================
# CALLBACK 6 : RÉINITIALISATION DE LA ZONE D'UPLOAD APRÈS CHARGEMENT
# ====================================================
@app.callback(
    Output('upload-div', 'children'),
    Input('upload-annotations', 'contents'),
    prevent_initial_call=True
)
def reset_upload(contents):
    """
    Réinitialise l'affichage de la zone d'upload dès qu'un fichier a été chargé.
    Cela permet à l'utilisateur de voir à nouveau l'invite pour charger un fichier si nécessaire.
    """
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


# ====================================================
# POINT D'ENTRÉE DE L'APPLICATION
# ====================================================
if __name__ == '__main__':
    # Lancement du serveur Dash en mode non-debug en production
    app.run(debug=False)