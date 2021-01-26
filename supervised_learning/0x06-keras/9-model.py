#!/usr/bin/env python3
"""save and load"""
import tensorflow.keras as k


def save_model(network, filename):
    """saves an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """loads an entire model"""
    return k.models.load_model(filename)
