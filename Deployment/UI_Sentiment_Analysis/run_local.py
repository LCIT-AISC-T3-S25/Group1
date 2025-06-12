from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from lime.lime_text import LimeTextExplainer
from utils import preprocess_text

app = Flask(__name__)

# === Load LSTM model and resources ===
with open("tokenizer.pkl", "rb") as f:
    lstm_tokenizer = pickle.load(f)
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

# === Load GRU model and resources ===
with open("gru_tokenizer.pkl", "rb") as f:
    gru_tokenizer = pickle.load(f)
gru_model = load_model("GRU_model.keras")
gru_maxlen = 50  # Set according to training
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

# LIME explanation (generic class names)
explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive", "Mild", "Strong"])

def predict_prob(texts, model, tokenizer, maxlen):
    sequences = [preprocess_text(t, tokenizer, maxlen)[0] for t in texts]
    padded = np.array(sequences)
    return model.predict(padded)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/sentiment')
def sentiment():
    return render_template("index.html")

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
