#!/usr/bin/env python3
"""creates a tensorflow layer that includes L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes L2 regularization"""
    kernel = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    la = tf.layers.Dense(units=n, activation=activation, name="layer",
                         kernel_initializer=init, kernel_regularizer=kernel)
    return la(prev)
