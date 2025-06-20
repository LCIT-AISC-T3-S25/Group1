from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
from lime.lime_text import LimeTextExplainer
from utils import preprocess_text  # ✅ Now importing from utils.py

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load maxlen
with open("maxlen.txt", "r") as f:
    maxlen = int(f.read().strip())

# Load GRU model
model = load_model("model/GRU_model.keras")

# Define label mapping
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
class_names = list(label_map.values())

# Initialize Flask app
app = Flask(__name__)

# LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

# Helper function for LIME predictions
def predict_prob(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return model.predict(padded)

@app.route("/")
def home():
    return "GRU Sentiment Analysis API with LIME is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data.get("tweet", "")
    if not tweet:
        return jsonify({"error": "No 'tweet' field provided in input"}), 400

    # Preprocess and predict
    padded_input = preprocess_text(tweet, tokenizer, maxlen)
    prediction = model.predict(padded_input)[0]

    class_index = int(np.argmax(prediction))
    predicted_class = label_map[class_index]

    class_probabilities = {
        label_map[i]: round(float(prob), 3) for i, prob in enumerate(prediction)
    }

    # LIME explanation
    exp = explainer.explain_instance(tweet, predict_prob, num_features=10, top_labels=1)
    lime_explanation = exp.as_list(label=exp.top_labels[0])

    return jsonify({
        "tweet": tweet,
        "predicted_class": predicted_class,
        "class_probabilities": class_probabilities,
        "lime_explanation": lime_explanation
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
