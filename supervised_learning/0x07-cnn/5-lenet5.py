#!/usr/bin/env python3
"""builds a modified version of the LeNet-5 architecture using keras"""
import tensorflow.keras as k


def lenet5(X):
    """builds a modified version of the LeNet-5 architecture using keras"""
    kernel = k.initializers.he_normal(seed=None)
    conv1 = k.layers.Conv2D(filters=6, kernel_size=(5, 5),
                            kernel_initializer=kernel,
                            padding="SAME", activation='relu')(X)

    pool1 = k.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = k.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            kernel_initializer=kernel,
                            *padding="VALID", activation='relu')(pool1)

    pool2 = k.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flat = k.layers.Flatten()(pool2)

    layer1 = k.layers.Dense(units=120,
                            kernel_initializer=kernel, activation='relu')(flat)

    layer2 = k.layers.Dense(units=84, kernel_initializer=kernel,
                            activation='relu')(layer1)

    layer3 = k.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=kernel)(layer2)

    model = k.models.Model(inputs=X, outputs=layer3)

    model.compile(optimizer=k.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
