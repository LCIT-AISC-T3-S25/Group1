import pickle
import numpy as np

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def preprocess_text(text: str) -> str:
    return text.strip().lower()

def tokenize_and_pad(texts, tokenizer, max_len):
    if hasattr(tokenizer, "texts_to_sequences"):
        sequences = tokenizer.texts_to_sequences(texts)
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    elif hasattr(tokenizer, "encode"):
        # For HF tokenizers
        encodings = [tokenizer.encode(t, truncation=True, max_length=max_len) for t in texts]
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return pad_sequences(encodings, maxlen=max_len, padding="post", truncating="post")
    else:
        raise ValueError("Unsupported tokenizer type")
