#!/usr/bin/env python3
"""builds a neural network with the Keras library"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = k.Sequential()
    k_r = k.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(k.layers.Dense(layers[i], input_dim=nx,
                      activation=activations[i], kernel_regularizer=k_r))
        else:
            model.add(k.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer=k_r))
        if i < len(layers) - 1:
            model.add(k.layers.Dropout(1-keep_prob))
    return model
