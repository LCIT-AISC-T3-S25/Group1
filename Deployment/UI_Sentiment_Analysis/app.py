from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json
import pickle
from lime.lime_text import LimeTextExplainer
from lime import lime_image
from PIL import Image
from torchvision import models
import torch
import base64
import io
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# === Import utilities ===
from utils.preprocess_text import preprocess_text
from utils.image_utils import preprocess_image
from interpret.gradcam import get_gradcam_image

app = Flask(__name__)

# === Load LSTM model and tokenizer from JSON ===
with open("tokenizer.json") as f:
    tokenizer_json = f.read()
lstm_tokenizer = tokenizer_from_json(tokenizer_json)

with open("maxlen.txt", "r") as f:
    lstm_maxlen = int(f.read().strip())
lstm_model = load_model("LSTM_model.h5")

label_map_lstm = {
    0: "Strong Negative",
    1: "Mild Negative",
    2: "Neutral",
    3: "Mild Positive",
    4: "Strong Positive"
}
emoji_map_lstm = {
    "Strong Negative": "😠",
    "Mild Negative": "🙁",
    "Neutral": "😐",
    "Mild Positive": "🙂",
    "Strong Positive": "😄"
}

# === Load GRU model and tokenizer from Pickle (assumed safe) ===
with open("gru_tokenizer.pkl", "rb") as f:
    gru_tokenizer = pickle.load(f)

gru_model = load_model("GRU_model.keras")
gru_maxlen = 50

label_map_gru = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
emoji_map_gru = {
    "Negative": "🙁",
    "Neutral": "😐",
    "Positive": "🙂"
}

explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive", "Mild", "Strong"])

def predict_prob(texts, model, tokenizer, maxlen):
    sequences = [preprocess_text(t, tokenizer, maxlen)[0] for t in texts]
    return model.predict(np.array(sequences))

# === Load Image Classification Models ===
vgg_model = models.vgg16(weights=None)
vgg_model.classifier[6] = torch.nn.Linear(4096, 5)
vgg_model.load_state_dict(torch.load("models/vgg_finetuned_round2.pth", map_location=torch.device("cpu")))
vgg_model.eval()

cnn_model = load_model("models/CNN_model_tuned2.keras")
class_names = ["drink", "food", "inside", "menu", "outside"]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/sentiment')
def sentiment():
    return render_template("index.html")

@app.route('/image-classification', methods=["GET", "POST"])
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'text' not in data or 'model' not in data:
        return jsonify({"error": "Missing 'text' or 'model' in request"}), 400

    text = data['text']
    model_type = data['model'].lower()

    if model_type == "gru":
        model = gru_model
        tokenizer = gru_tokenizer
        maxlen = gru_maxlen
        label_map = label_map_gru
        emoji_map = emoji_map_gru
    else:
        model = lstm_model
        tokenizer = lstm_tokenizer
        maxlen = lstm_maxlen
        label_map = label_map_lstm
        emoji_map = emoji_map_lstm

    padded = preprocess_text(text, tokenizer, maxlen)
    prediction = model.predict(padded)[0]
    class_index = int(np.argmax(prediction))
    predicted_class = label_map[class_index]
    predicted_emoji = emoji_map[predicted_class]

    class_probabilities = {
        label_map[i]: float(prediction[i])
        for i in range(len(prediction))
    }

    exp = explainer.explain_instance(
        text,
        lambda x: predict_prob(x, model, tokenizer, maxlen),
        num_features=10,
        top_labels=1
    )
    explanation = exp.as_list(label=exp.top_labels[0])

    return jsonify({
        "text": text,
        "predicted_class": predicted_class,
        "emoji": predicted_emoji,
        "probabilities": class_probabilities,
        "lime_explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
