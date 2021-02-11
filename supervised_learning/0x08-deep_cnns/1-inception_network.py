#!/usr/bin/env python3
"""builds the inception network as described in Going Deeper with
Convolutions (2014)"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """builds the inception network as described in Going Deeper
    with Convolutions (2014)"""
    X = K.Input(shape=(224, 224, 3))
    kernel = K.initializers.he_normal()

    l1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), padding='same',
                         activation='relu', kernel_initializer=kernel,
                         strides=(2, 2))(X)

    l2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(l1)

    l3 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                         activation='relu', kernel_initializer=kernel)(l2)

    l33 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same',
                          activation='relu', kernel_initializer=kernel)(l3)

    l4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(l33)

    l5 = inception_block(l4, [64, 96, 128, 16, 32, 32])

    l6 = inception_block(l5, [128, 128, 192, 32, 96, 64])

    l7 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(l6)

    l8 = inception_block(l7, [192, 96, 208, 16, 48, 64])

    l9 = inception_block(l8, [160, 112, 224, 24, 64, 64])

    l10 = inception_block(l9, [128, 128, 256, 24, 64, 64])

    l11 = inception_block(l10, [112, 144, 288, 32, 64, 64])

    l12 = inception_block(l11, [256, 160, 320, 32, 128, 128])

    l13 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                padding='same')(l12)

    l14 = inception_block(l13, [256, 160, 320, 32, 128, 128])

    l15 = inception_block(l14, [384, 192, 384, 48, 128, 128])

    l16 = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(l15)

    l17 = K.layers.Dropout(0.4)(l16)

    l18 = K.layers.Dense(activation='softmax', units=1000,
                         kernel_initializer=kernel)(l17)

    model = K.models.Model(inputs=X, outputs=l18)

    return model
