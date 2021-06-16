#!/usr/bin/env python3
"""class transformer"""
import tensorflow as tf
Decoder = __import__('10-transformer_decoder').Decoder
Encoder = __import__('9-transformer_encoder').Encoder


class Transformer(tf.keras.Model):
    """class transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """initialize class constructor"""
        super().__init__()

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """call function"""
        return 0
