#!/usr/bin/env python3
"""create all masks for training/validation"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_masks(inputs, target):
    """create all masks for training/validation"""
    batch_size, seq_len_in = inputs.shape
    batch_size, seq_len_out = target.shape

    # the encoder mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # the decoder mask
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # the look ahead mask
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((
        seq_len_out, seq_len_out)), -1, 0)

    #  the decoder target padding mask.
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[
        :, tf.newaxis, tf.newaxis, :]
    # combined_mask
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
