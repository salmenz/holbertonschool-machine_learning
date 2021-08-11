#!/usr/bin/env python3
"""randomly shears an image"""
import tensorflow as tf


def shear_image(image, intensity):
    """randomly shears an image"""
    img = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(img, intensity,
                                                        channel_axis=2)
    return tf.keras.preprocessing.image.array_to_img(sheared)
