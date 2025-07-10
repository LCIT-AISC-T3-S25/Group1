import os
import nbformat
from nbformat.v4 import new_notebook

# Correct path from your directory
folder_path = "NLP/case_study2/src/notebooks/"

# Your desired notebook merge order:
custom_order = [
    "NLP_Q4_subhash.ipynb",
    "NLP_Q4_sangeethFinal.ipynb"
]

# Initialize the combined notebook
combined_nb = new_notebook()
combined_cells = []

# Add contents of each notebook with a markdown header
for nb_file in custom_order:
    nb_path = os.path.join(folder_path, nb_file)
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            combined_cells.append(nbformat.v4.new_markdown_cell(f"# 📘 Notebook: {nb_file}"))
            combined_cells.extend(nb.cells)
    except FileNotFoundError:
        print(f"⚠️ WARNING: Notebook not found: {nb_file} — skipping.")

# Set final cell list and save the merged notebook
combined_nb.cells = combined_cells

output_path = os.path.join(folder_path, 'NLP_Q4_combine.ipynb')
os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists
with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(combined_nb, f)

print(f"✅ Combined notebook saved to: {output_path}")
