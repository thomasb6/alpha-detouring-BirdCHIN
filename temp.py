from dash import Dash, Input, Output, html, ALL, no_update, callback_context, _dash_renderer
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from pathlib import Path

_dash_renderer._set_react_version("18.2.0")


class FileTree:
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)

    def render(self) -> dmc.Accordion:
        return dmc.Accordion(
            children=self.build_tree(self.filepath, is_root=True),
            multiple=True
        )

    def flatten(self, input):
        return [item for sublist in input for item in sublist]

    def make_file(self, file_name):
        return dmc.Group([
            # Spacing element to align the checkbox with the file icon
            dmc.Space(w=35),
            DashIconify(icon="akar-icons:file"),
            dmc.Text(file_name)
        ], style={"paddingTop": '5px'})

    def make_folder(self, folder_name, path):
        return dmc.Group([
            dmc.Checkbox(id={'type': 'folder_checkbox', 'index': str(path)}),
            DashIconify(icon="akar-icons:folder"),
            dmc.Text(folder_name)
        ])

    def build_tree(self, path, is_root=False):
        d = []
        path = Path(path)
        if path.is_dir():
            children = self.flatten([self.build_tree(child)
                                     for child in path.iterdir()])
            if is_root and path != INITIAL_FOLDER:
                d.append(
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            self.make_folder('..', path.parent)),
                    ], value='..')
                )
            d.append(
                dmc.AccordionItem([
                    dmc.AccordionControl(
                        self.make_folder(path.name, path)),
                    dmc.AccordionPanel(children=children)
                ], value=str(path))
            )
        else:
            d.append(self.make_file(path.name))
        return d


INITIAL_FOLDER = Path('/Volumes/SURPLUS/dataset_birdshot/jpg')

app = Dash(__name__)


# Define the callback
@app.callback(
    Output('filetree_div', 'children'),
    Output('selected_folder_title', 'children'),
    Input({'type': 'folder_checkbox', 'index': ALL}, 'checked')
)
def update_output(checked_values):
    '''
    Update the file tree and selected folder title based on the checked boxes
    '''
    if checked_values is None:
        return 'No paths selected'

    # Extract the paths of the checked checkboxes
    checked_paths = [item['id']['index'] for item, checked in zip(
        callback_context.inputs_list[0], checked_values) if checked]

    if checked_paths:
        # Render a new FileTree with the selected folder as the root
        return FileTree(checked_paths[0]).render(), checked_paths[0]
    else:
        return no_update


# Add an output div to your layout
app.layout = dmc.MantineProvider(
    children=[
        html.Div(
            [
                html.H2(id='selected_folder_title', children=str(INITIAL_FOLDER)),
                html.Div(id='filetree_div', children=FileTree(INITIAL_FOLDER).render())
            ]
        )
    ]
)

if __name__ == '__main__':
    app.run(debug=True)