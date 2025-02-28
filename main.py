import json
import os

import dash
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, callback
from skimage import io

app = Dash(__name__)
server = app.server

folder_path = '/Users/thomasfoulonneau/PycharmProjects/alpha-detouring-BirdCHIN/optos_jpg'
filenames = os.listdir(folder_path)

# Layout Dash
app.layout = html.Div([
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': f, 'value': f} for f in filenames],
        placeholder='Sélectionnez un fichier à analyser'
    ),
    html.Div(id='output-text')
])
# Callback pour afficher le fichier sélectionné
@app.callback(
    dash.Output('output-text', 'children'),
    [dash.Input('file-dropdown', 'value')]
)
def display_selected_file(selected_filename):
    if selected_filename:
        try:
            img = io.imread(os.path.join(folder_path, selected_filename))
            fig = px.imshow(img)
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
                dcc.Markdown("Characteristics de la zone sélectionnée"),
                html.Pre(id="annotations-pre")
            ])
        except Exception as e:
            return f'Error loading file: {str(e)}'
    return 'No file selected'

# Callback to update annotations
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

# Run the app
if __name__ == '__main__':
    app.run(debug=False)
