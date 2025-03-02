import json
import requests
import dash
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, callback
from skimage import io
import io as io_buffer
from PIL import Image

# Configuration GitHub (dépôt public)
REPO_OWNER = "thomasb6"
REPO_NAME = "alpha-detouring-BirdCHIN"
FOLDER_PATH = "optos_jpg"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FOLDER_PATH}"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Récupérer le token depuis Render

def get_filenames():
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(GITHUB_API_URL, headers=headers)
    print("GitHub API Response:", response.status_code, response.text)  # Debug
    if response.status_code == 200:
        return [file["name"] for file in response.json() if file["type"] == "file"]
    return []


def get_image_url(filename):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{FOLDER_PATH}/{filename}"

app = Dash(__name__)
server = app.server

filenames = get_filenames()

# Layout Dash
app.layout = html.Div([
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': f, 'value': f} for f in filenames],
        placeholder='Sélectionnez un fichier à analyser'
    ),
    html.Div(id='output-text')
])

@app.callback(
    Output('output-text', 'children'),
    [Input('file-dropdown', 'value')]
)
def display_selected_file(selected_filename):
    if selected_filename:
        try:
            image_url = get_image_url(selected_filename)
            response = requests.get(image_url)
            image = Image.open(io_buffer.BytesIO(response.content))
            fig = px.imshow(image)
            fig.update_layout(dragmode="drawclosedpath")
            config = {
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape",
                ]
            }
            return html.Div([
                html.H4("Vous pouvez contourer le fichier"),
                dcc.Graph(id="fig-image", figure=fig, config=config),
                dcc.Markdown("Caractéristiques de la zone sélectionnée"),
                html.Pre(id="annotations-pre")
            ])
        except Exception as e:
            return f'Error loading file: {str(e)}'
    return 'No file selected'

@callback(
    Output("annotations-pre", "children"),
    Input("fig-image", "relayoutData"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data):
    for key in relayout_data:
        if "shapes" in key:
            return json.dumps(f'{key}: {relayout_data[key]}', indent=2)
    return no_update

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8080)
