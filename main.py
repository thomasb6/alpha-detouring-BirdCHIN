import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, callback
from skimage import data
import json

img = data.chelsea()
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

# Build App
app = Dash()
app.layout = html.Div(
    [
        html.H4("Draw a shape, then modify it"),
        dcc.Graph(id="fig-image", figure=fig, config=config),
        dcc.Markdown("Characteristics of shapes"),
        html.Pre(id="annotations-pre"),
    ]
)

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

if __name__ == "__main__":
    app.run(debug=True)