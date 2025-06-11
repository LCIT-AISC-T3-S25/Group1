from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load model and tokenizer
model = load_model("LSTM_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Class labels (in same order as model output)
label_map = {
    0: "Mild Negative",
    1: "Mild Positive",
    2: "Neutral",
    3: "Strong Negative",
    4: "Strong Positive"
}

# Set to the same value used during training
MAX_LEN = 50  # replace with your actual max_len if different

@app.route('/')
def index():
    return "LSTM Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    if 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data['text']

    # Tokenize and pad the input
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

    # Predict class index
    prediction = model.predict(padded)
    class_index = int(np.argmax(prediction))
    predicted_class = label_map[class_index]

    return jsonify({
        "predicted_class": predicted_class
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
