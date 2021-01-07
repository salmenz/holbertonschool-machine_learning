#!/usr/bin/env python3
"""creates the training operation for the network"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    lo = loss
    return tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(lo)
