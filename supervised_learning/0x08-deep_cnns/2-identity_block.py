#!/usr/bin/env python3
"""builds an identity block as described in Deep
Residual Learning for Image Recognition (2015)"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """builds an identity block as described in Deep
    Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters
    kernel = K.initializers.he_normal()

    l1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel,
                         strides=1)(A_prev)

    b1 = K.layers.BatchNormalization()(l1)

    d = K.layers.Activation('relu')(b1)

    l2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                         activation='relu', kernel_initializer=kernel,
                         strides=1)(d)

    b2 = K.layers.BatchNormalization()(l2)

    d2 = K.layers.Activation('relu')(b2)

    l3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel,
                         strides=1)(d2)

    b3 = K.layers.BatchNormalization()(l3)

    l4 = K.layers.Add()([b3, A_prev])

    out = K.layers.Activation('relu')(l4)

    return out
