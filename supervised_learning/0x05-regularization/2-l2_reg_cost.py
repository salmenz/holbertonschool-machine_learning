#!/usr/bin/env python3
"""creates a tensorflow layer that includes L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """creates a tensorflow layer that includes L2 regularization"""
    return tf.losses.get_regularization_losses() + cost
