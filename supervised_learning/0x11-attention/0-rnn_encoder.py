#!/usr/bin/env python3
"""class RNNEncoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """class RNNEncoder"""
    def __init__(self, vocab, embedding, units, batch):
        """Class constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units, return_sequences=True,
                                       recurrent_initializer="glorot_uniform",
                                       return_state=True)

    def initialize_hidden_state(self):
        """call function"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """call function"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
