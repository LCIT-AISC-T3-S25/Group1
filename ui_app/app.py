# --- Import necessary libraries ---
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Import libraries for NLP models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from lime.lime_text import LimeTextExplainer
from lime import lime_image

# Import libraries for Image Classification models
import torch
import base64
import io
from PIL import Image
from torchvision import models
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Import user-defined utilities
from utils.preprocess_text import clean_text, handle_negation
from utils.image_utils import preprocess_image
from interpret.gradcam import get_gradcam_image

app = Flask(__name__)

# --- Load Utils and Models ---

# Load Tokenizer and Encoder for pre-processing Text
with open("models/Pre-processing/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("models/Pre-processing/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Shared maxlen for both models
MAXLEN = 62

# Load NLP models
gru_model = load_model("models/GRU_model.keras")
lstm_model = load_model("models/LSTM_model.keras")

# Load Image Classification models
cnn_model = load_model("models/CNN_model.keras")
vgg_model = models.vgg16(weights=None)
vgg_model.classifier[6] = torch.nn.Linear(4096, 5)
vgg_model.load_state_dict(torch.load("models/VGG_model.pth", map_location=torch.device("cpu")))
vgg_model.eval()

class_names = ["drink", "food", "inside", "menu", "outside"]

# Label and emoji mappings
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
emoji_map = {
    "Negative": "🙁",
    "Neutral": "😐",
    "Positive": "🙂"
}

# Initialize Lime for interpretability (NLP Models)
explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])

def predict_prob(texts, model, tokenizer, maxlen):
    cleaned_texts = [handle_negation(clean_text(t)) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    # Fallback for empty texts
    sequences = [[0] if not s else s for s in sequences]
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post')
    preds = model.predict(padded)
    return preds


@app.route('/')
def home():
    return render_template("homePage.html")

@app.route('/sentiment')
def sentiment():
    return render_template("sentimentPage.html")

@app.route('/predict', methods=['POST'])
# Define function to predict for NLP models
def predict():
    data = request.get_json(force=True)
    if 'text' not in data or 'model' not in data:
        return jsonify({"error": "Missing 'text' or 'model' in request"}), 400

    text = data['text']
    model_type = data['model'].lower()

    cleaned = clean_text(text)
    negated = handle_negation(cleaned)
    seq = tokenizer.texts_to_sequences([negated])
    if not seq[0]:
        seq[0] = [0]  # handle empty sequence
    padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')

    if model_type == "gru":
        prediction = gru_model.predict(padded)[0]
        model = gru_model
    else:
        prediction = lstm_model.predict(padded)[0]
        model = lstm_model

    class_index = int(np.argmax(prediction))
    predicted_class = label_map[class_index]
    predicted_emoji = emoji_map[predicted_class]

    class_probabilities = {
        label_map[i]: float(prediction[i]) for i in range(len(prediction))
    }

    exp = explainer.explain_instance(
        text,
        lambda x: predict_prob(x, model, tokenizer, MAXLEN),
        num_features=10,
        top_labels=1,
        num_samples=50

    )
    explanation = exp.as_list(label=exp.top_labels[0])

    return jsonify({
        "text": text,
        "predicted_class": predicted_class,
        "emoji": predicted_emoji,
        "probabilities": class_probabilities,
        "lime_explanation": explanation
    })

@app.route('/image-classification', methods=["GET", "POST"])
# Define function for image classification
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

            lime_explainer = lime_image.LimeImageExplainer()

            def predict_fn(images): return cnn_model.predict(np.array(images))

            explanation = lime_explainer.explain_instance(
                img_array[0], predict_fn, top_labels=1, hide_color=0, num_samples=100
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

    return render_template("imagePage.html",
                           prediction=prediction,
                           confidence=confidence,
                           heatmap=heatmap_b64,
                           uploaded=uploaded_b64,
                           model_selected=selected_model,
                           error=error)


if __name__ == "__main__":
    app.run(debug=True)
