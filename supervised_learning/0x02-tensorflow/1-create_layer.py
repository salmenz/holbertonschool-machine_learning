#!/usr/bin/env python3
"""create layer with tensorflow"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """create layer with tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    la = tf.layers.dense(prev, units=n, activation=activation, name="layer",
                         kernel_initializer=kernel, reuse=tf.AUTO_REUSE)
    return layer(la)
