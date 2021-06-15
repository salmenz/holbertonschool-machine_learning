#!/usr/bin/env python3
"""calculate the scaled dot product attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """calculate the scaled dot product attention"""
    qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = qk / tf.math.sqrt(dk)
    if mask:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights
