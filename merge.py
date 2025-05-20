import os
import nbformat
from nbformat.v4 import new_notebook


folder_path = 'Case_Study1/src/notebook'


notebook_files = [f for f in os.listdir(folder_path) if f.endswith('.ipynb') and not f.startswith('.')]

notebook_files.sort()


combined_nb = new_notebook()
combined_cells = []


for nb_file in notebook_files:
    nb_path = os.path.join(folder_path, nb_file)
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
        combined_cells.append(nbformat.v4.new_markdown_cell(f"# Notebook: {nb_file}"))
        combined_cells.extend(nb.cells)


combined_nb.cells = combined_cells


output_path = os.path.join(folder_path, 'Combined_Notebook.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(combined_nb, f)

print(f"Combined notebook saved to: {output_path}")
