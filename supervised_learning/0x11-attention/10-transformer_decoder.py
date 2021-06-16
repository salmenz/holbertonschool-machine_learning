#!/usr/bin/env python3
"""class Decoder"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """class Decoder"""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """class constructor"""
        super().__init__()

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """call function"""
        return 0
