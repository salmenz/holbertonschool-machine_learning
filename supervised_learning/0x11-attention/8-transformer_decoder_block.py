#!/usr/bin/env python3
"""class DecoderBlock"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """class DecoderBlock"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """class constructor """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """call function"""
        att, att_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        att = self.dropout1(att, training=training)
        output = self.layernorm1(att + x)
        att2, att_weights_block2 = self.mha2(output, encoder_output,
                                            encoder_output, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(att2 + output)
        out = self.dense_hidden(out2)
        out = self.dense_output(out)
        out = self.dropout3(out, training=training)
        return self.layernorm3(out + out2)
