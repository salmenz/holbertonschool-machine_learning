#!/usr/bin/env python3
"""class MultiHeadAttention"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """class MultiHeadAttention"""
    def __init__(self, dm, h):
        """class constructor"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        assert dm % self.h == 0
        self.depth = dm // self.h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def splitt(self, x, batch_size):
        """Split the last dimension"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """call function"""
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.splitt(q, batch_size)
        k = self.splitt(k, batch_size)
        v = self.splitt(v, batch_size)
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        con_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))
        output = self.linear(con_attention)
        return output, attention_weights