import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim, max_len, embedding_matrix=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        if embedding_matrix is not None:
            self.embedding_matrix = np.array(embedding_matrix)
        else:
            self.embedding_matrix = np.zeros((vocab_size, embedding_dim))
        self.token_embed = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[self.embedding_matrix] if embedding_matrix is not None else None,
            trainable=False
        )
        self.pos_embed = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embedding_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_embed(positions)
        x = self.token_embed(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_len': self.max_len,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop('embedding_matrix', None)
        return cls(**config)
