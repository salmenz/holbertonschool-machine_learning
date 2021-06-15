#!/usr/bin/env python3
"""class SelfAttention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """class SelfAttention"""
    def __init__(self, units):
        """Class constructor"""
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """call function"""
        prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
