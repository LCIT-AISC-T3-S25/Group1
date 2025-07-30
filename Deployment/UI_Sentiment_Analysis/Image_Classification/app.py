from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import models
import numpy as np
import base64
import io
from utils.image_utils import preprocess_image
from interpret.gradcam import get_gradcam_image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load models
vgg_model = models.vgg16(weights=None)
vgg_model.classifier[6] = torch.nn.Linear(4096, 5)
vgg_model.load_state_dict(torch.load("models/vgg_finetuned_round2.pth", map_location=torch.device("cpu"), weights_only=True))
vgg_model.eval()

cnn_model = load_model("models/CNN_model_tuned2.keras")

class_names = ["drink", "food", "inside", "menu", "outside"]

@app.route("/")
def home():
    return render_template("image_ui.html")

@app.route("/image", methods=["GET", "POST"])
def image_classification():
    prediction = None
    confidence = None
    heatmap_b64 = None
    uploaded_b64 = None
    selected_model = None
    error = None

    if request.method == 'POST':
        selected_model = request.form.get("model")

        if 'image' not in request.files or request.files['image'].filename == '':
            error = "No image uploaded."
            return render_template('image_ui.html', error=error)

        file = request.files['image']
        image = Image.open(file).convert("RGB")

        buffer_img = io.BytesIO()
        image.save(buffer_img, format="PNG")
        uploaded_b64 = base64.b64encode(buffer_img.getvalue()).decode()

        if selected_model == "vgg16":
            tensor = preprocess_image(image)
            with torch.no_grad():
                output = vgg_model(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                pred_idx = torch.argmax(probs).item()
                prediction = class_names[pred_idx]
                confidence = round(probs[0][pred_idx].item(), 3)

            heatmap = get_gradcam_image(vgg_model, tensor)
            heatmap = Image.fromarray(heatmap.astype(np.uint8))
        
            buf = io.BytesIO()
            heatmap.save(buf, format="PNG")
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

        elif selected_model == "cnn":
            img_array = image.resize((128, 128))
            img_array = img_to_array(img_array) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction_arr = cnn_model.predict(img_array)
            pred_idx = np.argmax(prediction_arr[0])
            prediction = class_names[pred_idx]
            confidence = round(float(np.max(prediction_arr[0])), 3)

            explainer = lime_image.LimeImageExplainer()

            def predict_fn(images):
                return cnn_model.predict(np.array(images))

            explanation = explainer.explain_instance(
                img_array[0], predict_fn, top_labels=1, hide_color=0, num_samples=1000
            )
            lime_img, mask = explanation.get_image_and_mask(
                label=explanation.top_labels[0], positive_only=True,
                hide_rest=False, num_features=5
            )

            plt.imshow(mark_boundaries(lime_img, mask))
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode()
            plt.close()
        
        else:
            error = "Please select a model."

    return render_template("image_ui.html",
                           prediction=prediction,
                           confidence=confidence,
                           heatmap=heatmap_b64,
                           uploaded=uploaded_b64,
                           model_selected=selected_model,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
