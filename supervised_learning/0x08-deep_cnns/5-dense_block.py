#!/usr/bin/env python3
"""builds a dense block as described in Densely
Connected Convolutional Networks"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block as described in Densely
    Connected Convolutional Networks"""
    out = X
    kernel = K.initializers.he_normal()
    for i in range(layers):
        dence_f = K.layers.BatchNormalization()(out)
        dence_f = K.layers.ReLU()(dence_f)
        inter_ch = growth_rate * 4
        dence_f = K.layers.Conv2D(inter_ch, kernel_size=(1, 1), padding='same',
                                  kernel_initializer=kernel,)(dence_f)

        dence_f = K.layers.BatchNormalization()(dence_f)
        dence_f = K.layers.ReLU()(dence_f)
        dence_f = K.layers.Conv2D(growth_rate, kernel_size=(3, 3),
                                  padding='same',
                                  kernel_initializer=kernel)(dence_f)
        out = K.layers.concatenate([out, dence_f])
        nb_filters += growth_rate
    return out, nb_filters
