#!/usr/bin/env python3
"""Transformer Encoder"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encoder class"""
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """initialization Class constructor"""
        super(Encoder, self).__init__()

    def call(self, x, training, mask):
        """call function"""
        return 0
