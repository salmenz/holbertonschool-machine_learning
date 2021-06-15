#!/usr/bin/env python3
"""class RNNDecoder"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.Model):
    """class RNNDecoder"""
    def __init__(self, vocab, embedding, units, batch):
        """class constructor"""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """call function"""
        context_v = self.attention(s_prev, hidden_states)[0]
        context_v = tf.expand_dims(context_v, 1)
        x = self.embedding(x)
        x = tf.concat([context_v, x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.F(output)
        return x, state
