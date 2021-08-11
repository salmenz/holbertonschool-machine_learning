#!/usr/bin/env python3
"""perform a random crop of an image"""
import tensorflow as tf


def crop_image(image, size):
    """perform a random crop of an image"""
    return tf.image.random_crop(image, size)
