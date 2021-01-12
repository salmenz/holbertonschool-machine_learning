#!/usr/bin/env python3
"""updates a variable using the RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """updates a variable using the RMSProp optimization algorithm"""
    return tf.train.RMSPropOptimizer(alpha, beta2,
                                     epsilon=epsilon).minimize(loss)
