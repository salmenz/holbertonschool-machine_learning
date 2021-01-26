#!/usr/bin/env python3
"""sets up Adam optimization for a keras model with
categorical crossentropy loss and accuracy metrics"""
import tensorflow.keras as k


def optimize_model(network, alpha, beta1, beta2):
    """sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics"""
    op = k.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy', optimizer=op,
                    metrics=['accuracy'])
    return None
