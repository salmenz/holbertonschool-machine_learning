#!/usr/bin/env python3
"""andomly changes the brightness of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """andomly changes the brightness of an image:"""
    return tf.image.adjust_brightness(image, max_delta)
