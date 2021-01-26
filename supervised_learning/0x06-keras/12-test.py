#!/usr/bin/env python3
"""tests a neural network"""
import tensorflow.keras as k


def test_model(network, data, labels, verbose=True):
    """tests a neural network"""
    return network.evaluate(x=data, y=labels, verbose=verbose)
