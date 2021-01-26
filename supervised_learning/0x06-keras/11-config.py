#!/usr/bin/env python3
"""saves and loads a model’s weights"""
import tensorflow.keras as k


def save_config(network, filename):
    """saves a model’s weights"""
    with open(filename, "w") as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """loads a model’s weights"""
    with open(filename, "r") as f:
        read = f.read()
    return k.models.model_from_json(read)
