#!/usr/bin/env python3
"""returns two placeholders, x and y"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """ tf.placeholder"""
    x = tf.placeholder(tf.float32, [None, nx], name="x")
    y = tf.placeholder(tf.float32, [None, classes], name="y")
    return x, y
