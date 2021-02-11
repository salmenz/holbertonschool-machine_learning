#!/usr/bin/env python3
"""builds a dense block as described in Densely
Connected Convolutional Networks"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block as described in Densely
    Connected Convolutional Networks"""
    kernel = K.initializers.he_normal()
    for i in range(layers):
        dense_f = K.layers.BatchNormalization()(X)
        dense_f = K.layers.Activation('relu')(dense_f)
        inter_chan = growth_rate * 4
        dense_f = K.layers.Conv2D(inter_chan, (1, 1), padding='same',
                                  kernel_initializer=kernel)(dense_f)

        dense_f = K.layers.BatchNormalization()(dense_f)
        dense_f = K.layers.Activation('relu')(dense_f)
        dense_f = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                                  kernel_initializer=kernel)(dense_f)

        out = K.layers.concatenate([X, dense_f])
        nb_filters += growth_rate
    return out, nb_filters
