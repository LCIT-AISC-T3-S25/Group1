import os
import nbformat
from nbformat.v4 import new_notebook

# Correct path from your directory
folder_path = 'Computer_Vision/Assignment1/src/notebooks'

# Your desired notebook merge order:
custom_order = [
    'EfficientNet_model_tuned.ipynb',
    'Q1. Final_Tuned_model1_custom_Lyrs_VGG_TL_.ipynb',
    'CV_Assignment1_question2.ipynb',
    
    'ques_4_vgg_Model.ipynb',
    'Yelp_Images_VGG16_TransferLearning.ipynb',
    'Q.5_vgg_modeltuningcode.ipynb'
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

output_path = os.path.join(folder_path, 'AISC2008_CV_Assignment1-Group1.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(combined_nb, f)

print(f"✅ Combined notebook saved to: {output_path}")
