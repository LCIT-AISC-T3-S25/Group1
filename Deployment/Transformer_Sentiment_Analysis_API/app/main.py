import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from .inference import predict, get_class_names
from .lime_explainer import explain_text
from .config import DEFAULT_PORT, DEFAULT_NUM_FEATURES, DEFAULT_NUM_SAMPLES

PORT = int(os.getenv("PORT", DEFAULT_PORT))
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({"classes": get_class_names()}), 200

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    texts = [data["text"]] if isinstance(data, dict) and "text" in data else data.get("texts", [])
    if not texts:
        return jsonify({"error": "Provide 'text' or 'texts'."}), 400
    results = predict(texts)
    return jsonify({"results": results}), 200

@app.route("/explain", methods=["POST"])
def explain_route():
    from traceback import print_exc
    try:
        data = request.get_json(force=True)
        text = data.get("text", None)
        if not text:
            return jsonify({"error": "Provide 'text'."}), 400
        num_features = int(data.get("num_features", DEFAULT_NUM_FEATURES))
        num_samples = int(data.get("num_samples", DEFAULT_NUM_SAMPLES))
        target_class = data.get("target_class", None)
        if target_class is not None:
            target_class = int(target_class)
        explanation = explain_text(
            text,
            num_features=num_features,
            num_samples=num_samples,
            target_class=target_class
        )
        return jsonify(explanation), 200
    except Exception as e:
        print("EXCEPTION in /explain:", e)
        print_exc()
        return jsonify({"error": str(e)}), 500


def create_app():
    return app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
