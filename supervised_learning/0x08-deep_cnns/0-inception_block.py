#!/usr/bin/env python3
"""builds an inception block as described in Going
Deeper with Convolutions (2014)"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """builds an inception block as described in Going
    Deeper with Convolutions (2014)"""
    F1, F3R, F3, F5R, F5, FPP = filters
    kernel = K.initializers.he_normal()
    l1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel)(A_prev)

    l2 = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel)(A_prev)

    l2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                         activation='relu', kernel_initializer=kernel)(l2)

    l3 = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel)(A_prev)

    l3 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                         activation='relu', kernel_initializer=kernel)(l3)

    l4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                               padding='same')(A_prev)

    l4 = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel)(l4)

    out = K.layers.concatenate([l1, l2, l3, l4])

    return out
