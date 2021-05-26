#!/usr/bin/env python3
import tensorflow as tf
"""class RNNEncoder"""


class RNNEncoder(tf.keras.layers.Layer):
    """class RNNEncoder"""
    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(output_dim=embedding,
                                                   input_dim=vocab)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       kernel_initializer="glorot_uniform",
                                       return_state=True)

    def initialize_hidden_state(self):
        """call function"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """call function"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
