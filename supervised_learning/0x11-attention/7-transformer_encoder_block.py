#!/usr/bin/env python3
"""class EncoderBlock"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """class EncoderBlock"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """class constructor"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """call function"""
        att_out, _ = self.mha(x, x, x, mask)
        att_out = self.dropout1(att_out, training=training)
        out1 = self.layernorm1(x + att_out)
        output = self.dense_hidden(out1)
        output = self.dense_output(output)
        output = self.dropout2(output, training=training)
        return self.layernorm2(out1 + output)
