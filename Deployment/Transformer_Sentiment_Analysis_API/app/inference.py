
import os
import pickle
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from .transformer_layers import PositionalEmbedding

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "transformer_sentiment_model.keras")
DEFAULT_TOKENIZER_PATH = os.path.join(DEFAULT_MODELS_DIR, "tokenizer.pkl")
DEFAULT_LABEL_MAP_PATH = os.path.join(DEFAULT_MODELS_DIR, "label_map.pkl")
DEFAULT_PARAMS_PATH = os.path.join(DEFAULT_MODELS_DIR, "model_params.pkl")

class SentimentPredictor:
    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
                 label_map_path: str = DEFAULT_LABEL_MAP_PATH,
                 params_path: str = DEFAULT_PARAMS_PATH):
        with open(params_path, 'rb') as f:
            self.params = pickle.load(f)
        self.max_len = self.params.get("maxlen", self.params.get("max_len", 128))
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(label_map_path, 'rb') as f:
            self.label_map = pickle.load(f)
        self.idx_to_name = self._build_idx_to_name(self.label_map)
        self.model = load_model(model_path, compile=False,
                                custom_objects={'PositionalEmbedding': PositionalEmbedding})

    @staticmethod
    def _build_idx_to_name(label_map):
        if isinstance(label_map, dict):
            first_key = next(iter(label_map.keys()))
            return label_map if isinstance(first_key, int) else {v: k for k, v in label_map.items()}
        elif isinstance(label_map, (list, tuple)):
            return {i: name for i, name in enumerate(label_map)}
        else:
            raise TypeError("label_map.pkl must be dict or list/tuple.")

    def preprocess_texts(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

    def predict_proba_batch(self, texts: List[str]) -> np.ndarray:
        X = self.preprocess_texts(texts)
        return self.model.predict(X, verbose=0)

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        predictions = self.predict_proba_batch(texts)
        results = []
        for i, text in enumerate(texts):
            probs = predictions[i]
            pred_idx = int(np.argmax(probs))
            results.append({
                "text": text,
                "pred_label_id": pred_idx,
                "pred_label": self.idx_to_name[pred_idx],
                "probs": probs.astype(float).tolist()
            })
        return results

_predictor_singleton: Optional[SentimentPredictor] = None

def _get_predictor() -> SentimentPredictor:
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = SentimentPredictor()
    return _predictor_singleton

def predict(texts: List[str]) -> List[Dict[str, Any]]:
    return _get_predictor().predict_batch(texts)

def predict_proba(texts: List[str]) -> np.ndarray:
    return _get_predictor().predict_proba_batch(texts)

def get_class_names() -> List[str]:
    predictor = _get_predictor()
    return [predictor.idx_to_name[i] for i in sorted(predictor.idx_to_name.keys())]
