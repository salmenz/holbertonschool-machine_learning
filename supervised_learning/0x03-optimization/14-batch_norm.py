#!/usr/bin/env python3
"""creates a batch normalization layer for a neural network in tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates batch normalization layer for a neural network in tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, name="layer",
                            kernel_initializer=kernel)
    mean, variance = tf.nn.moments(layer(prev), [0])
    epsilon = 1e-8
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    Bnorm = tf.nn.batch_normalization(layer(prev), mean, variance, beta,
                                      gamma, epsilon)
    return activation(Bnorm)
