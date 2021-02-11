#!/usr/bin/env python3
"""Transition Layer builds a transition layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Transition Layer builds a transition layer"""
    kernel = K.initializers.he_normal()
    filters = nb_filters * compression
    out = K.layers.BatchNormalization()(X)
    out = K.layers.Activation('relu')(out)
    out = K.layers.Conv2D(int(filters), kernel_size=(1, 1), padding="same",
                          kernel_initializer=kernel)(out)
    out = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(out)
    return out, int(filters)