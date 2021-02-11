#!/usr/bin/env python3
"""builds the ResNet-50 architecture as described in Deep Residual
Learning for Image Recognition (2015)"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015)"""
    X = K.Input(shape=(224, 224, 3))
    kernel = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), padding='same',
                            kernel_initializer=kernel, strides=(2, 2))(X)

    conv1 = K.layers.BatchNormalization()(conv1)

    conv1 = K.layers.Activation('relu')(conv1)

    conv2_x = K.layers.MaxPool2D(pool_size=(3, 3),
                                 strides=2, padding='same')(conv1)

    conv2_x = projection_block(conv2_x, [64, 64, 256], 1)
    conv2_x = identity_block(conv2_x, [64, 64, 256])
    conv2_x = identity_block(conv2_x, [64, 64, 256])

    conv3_x = projection_block(conv2_x, [128, 128, 512])
    conv3_x = identity_block(conv3_x, [128, 128, 512])
    conv3_x = identity_block(conv3_x, [128, 128, 512])
    conv3_x = identity_block(conv3_x, [128, 128, 512])

    conv4_x = projection_block(conv3_x, [256, 256, 1024])
    conv4_x = identity_block(conv4_x, [256, 256, 1024])
    conv4_x = identity_block(conv4_x, [256, 256, 1024])
    conv4_x = identity_block(conv4_x, [256, 256, 1024])
    conv4_x = identity_block(conv4_x, [256, 256, 1024])
    conv4_x = identity_block(conv4_x, [256, 256, 1024])

    conv5_x = projection_block(conv4_x, [512, 512, 2048])
    conv5_x = identity_block(conv5_x, [512, 512, 2048])
    conv5_x = identity_block(conv5_x, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(conv5_x)

    s_max = K.layers.Dense(activation='softmax', units=1000,
                           kernel_initializer=kernel)(avg_pool)

    model = K.models.Model(inputs=X, outputs=s_max)

    return model
