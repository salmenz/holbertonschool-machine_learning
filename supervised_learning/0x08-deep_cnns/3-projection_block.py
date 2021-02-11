#!/usr/bin/env python3
"""hat builds a projection block as described in Deep Residual
Learning for Image Recognition (2015)"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """hat builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters
    kernel = K.initializers.he_normal()

    l11 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                          activation='relu', kernel_initializer=kernel,
                          strides=s)(A_prev)

    b11 = K.layers.BatchNormalization()(l11)

    l1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel,
                         strides=s)(A_prev)

    b1 = K.layers.BatchNormalization()(l1)

    d = K.layers.Activation('relu')(b1)

    l2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                         activation='relu', kernel_initializer=kernel)(d)

    b2 = K.layers.BatchNormalization()(l2)

    d2 = K.layers.Activation('relu')(b2)

    l3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel)(d2)

    b3 = K.layers.BatchNormalization()(l3)

    l4 = K.layers.Add()([b3, b11])

    out = K.layers.Activation('relu')(l4)

    return out
