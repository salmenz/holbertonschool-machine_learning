#!/usr/bin/env python3
"""rotate an image by 90 degrees counter-clockwise"""
import tensorflow as tf


def rotate_image(image):
    """rotate an image by 90 degrees counter-clockwise"""
    return tf.image.rot90(image)
