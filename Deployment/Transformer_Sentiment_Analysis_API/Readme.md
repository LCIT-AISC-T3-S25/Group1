{
  "model_name": "Transformer Sentiment Analyzer with LIME",
  "model_type": "Text Classification (Sentiment Analysis)",
  "version": "1.0.0",
  "description": "A Transformer-based sentiment analysis model with LIME interpretability, deployed using Flask and Docker.",
  "docker_info": {
    "image_name": "sentiment-analyzer",
    "port": 8001,
    "health_endpoint": "/health",
    "predict_endpoint": "/predict",
    "explain_endpoint": "/explain"
  },
  "api_endpoints": {
    "/health": {
      "method": "GET",
      "description": "Health check endpoint. Confirms service is running.",
      "response_format": {
        "status": "string"
      }
    },
    "/predict": {
      "method": "POST",
      "description": "Predict the sentiment of the provided text input.",
      "input_format": {
        "text": "string (required) - A single input text or",
        "texts": "list of strings (optional) - Multiple input texts"
      },
      "response_format": {
        "results": [
          {
            "text": "string - Input text",
            "pred_label_id": "integer - Class index",
            "pred_label": "string - Class name",
            "probs": "list[float] - Probabilities for all classes"
          }
        ]
      },
      "example_request": {
        "text": "I love this product!"
      },
      "example_response": {
        "results": [
          {
            "text": "I love this product!",
            "pred_label_id": 2,
            "pred_label": "Positive",
            "probs": [0.01, 0.05, 0.94]
          }
        ]
      }
    },
    "/explain": {
      "method": "POST",
      "description": "Explain the prediction using LIME. Returns top features influencing prediction.",
      "input_format": {
        "text": "string (required)",
        "num_features": "integer (optional, default=10)",
        "num_samples": "integer (optional, default=500)",
        "target_class": "integer (optional) - Class index to explain (default: predicted class)"
      },
      "response_format": {
        "text": "string - Original input text",
        "target_class_id": "integer - Index of explained class",
        "target_class_name": "string - Class name",
        "weights": [
          {
            "term": "string - Word or token",
            "weight": "float - Importance weight"
          }
        ]
      },
      "example_request": {
        "text": "I love this product!",
        "num_features": 5,
        "num_samples": 500
      },
      "example_response": {
        "text": "I love this product!",
        "target_class_id": 2,
        "target_class_name": "Positive",
        "weights": [
          {"term": "love", "weight": 0.215},
          {"term": "product", "weight": 0.103},
          {"term": "!", "weight": 0.045}
        ]
      }
    }
  },
  "technical_details": {
    "architecture": "Custom Transformer Encoder with Positional Embeddings",
    "explainability": "LIME Text Explainer",
    "framework": "TensorFlow + Flask",
    "model_format": "Saved Keras model (.h5)",
    "max_sequence_length": 100,
    "embedding_dim": 64,
    "num_heads": 2,
    "ff_dim": 64,
    "dropout": 0.1,
    "classes": ["Negative", "Neutral", "Positive"]
  },
  "deployment": {
    "framework": "Flask",
    "python_version": "3.11",
    "container_port": 8001,
    "health_check_interval": "10s",
    "startup_time": "5-10s"
  },
  "usage_instructions": {
    "build_image": "docker build -t sentiment-analyzer .",
    "run_container": "docker run -p 8001:8001 sentiment-analyzer",
    "test_health": "curl http://localhost:8001/health",
    "test_prediction": "curl -X POST http://localhost:8001/predict -H 'Content-Type: application/json' -d '{\"text\": \"I love this product!\"}'",
    "test_explanation": "curl -X POST http://localhost:8001/explain -H 'Content-Type: application/json' -d '{\"text\": \"I love this product!\", \"num_features\": 5, \"num_samples\": 500}'"
  },
  "requirements": [
    "flask==2.3.3",
    "flask-cors==4.0.0",
    "gunicorn==21.2.0",
    "tensorflow==2.15.0",
    "lime==0.2.0.1",
    "numpy==1.24.4",
    "scikit-learn==1.3.2"
  ],
  "performance": {
    "average_response_time": "1 second",
    "memory_usage": "Low (< 1GB)",
    "cpu_usage": "Low",
    "concurrent_requests": "10+"
  },
  "limitations": {
    "domain_specific": "General sentiment only, not domain-tuned",
    "context_length": "Limited to 100 tokens",
    "explainability": "LIME approximates, not exact reasoning"
  }
}
