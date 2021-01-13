#!/usr/bin/env python3
"""creates a batch normalization layer for a neural network in tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer for a neural network in tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, name="layer",
                         kernel_initializer=kernel)
    gamma = np.zeros(1, n)
    beta = np.ones(1, n)
    epsilon = 1e-8
    mean = np.mean(prev, axis=0)
    variance = np.var((prev - mean), axis=0)
    prev = (prev - mean) / (np.sqrt(variance + epsilon))
    return gamma * prev + beta
