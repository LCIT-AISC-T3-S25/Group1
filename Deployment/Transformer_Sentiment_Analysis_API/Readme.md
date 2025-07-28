# Transformer Sentiment Analyzer with LIME

## Model Overview
- **Model Name**: Transformer Sentiment Analyzer with LIME Explanations
- **Type**: Text Classification (Sentiment Analysis)
- **Version**: 1.0.0
- **Description**: A Transformer-based sentiment model with LIME explanations deployed via Flask and Docker.

## Docker Info
- **Image Name**: `sentiment-analyzer`
- **Port**: `8001`
- **Endpoints**:
  - `/health` – GET
  - `/predict` – POST
  - `/explain` – POST

## API Endpoints

### `GET /health`
- **Description**: Basic health check.
- **Response**:
  ```json
  { "status": "ok" }
  ```

---

### `POST /predict`
- **Description**: Predict sentiment from one or more texts.
- **Input**:
  ```json
  { "text": "I love this product!" }
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "text": "I love this product!",
        "pred_label_id": 2,
        "pred_label": "Positive",
        "probs": [0.01, 0.05, 0.94]
      }
    ]
  }
  ```

---

### `POST /explain`
- **Description**: Get LIME explanation for input text.
- **Input**:
  ```json
  {
    "text": "I love this product!",
    "num_features": 5,
    "num_samples": 500
  }
  ```
- **Response**:
  ```json
  {
    "text": "I love this product!",
    "target_class_id": 2,
    "target_class_name": "Positive",
    "weights": [
      { "term": "love", "weight": 0.215 },
      { "term": "product", "weight": 0.103 },
      { "term": "!", "weight": 0.045 }
    ]
  }
  ```

---

## Technical Details
- **Transformer + Positional Embedding**
- Classes: Negative, Neutral, Positive
- TensorFlow + Flask
- Max tokens: 100

---

## Deployment
- **Framework**: Flask
- **Port**: 8001
- **Python**: 3.11

## Usage

```bash
docker build -t sentiment-analyzer .
docker run -p 8001:8001 sentiment-analyzer
curl http://localhost:8001/health
curl -X POST http://localhost:8001/predict -H 'Content-Type: application/json' -d '{"text": "I love this product!"}'
curl -X POST http://localhost:8001/explain -H 'Content-Type: application/json' -d '{"text": "I love this product!", "num_features": 5, "num_samples": 500}'
```

---

## Requirements
- flask==2.3.3
- flask-cors==4.0.0
- tensorflow==2.15.0
- lime==0.2.0.1
- numpy, scikit-learn, gunicorn

---

## Performance & Limitations
- **Latency**: ~1 sec
- **Memory**: < 1 GB
- **Limitations**:
  - General sentiment only
  - LIME explanations are approximations
  - Max 100 tokens input
