#!/usr/bin/env python3
"""LeNet-5 (Tensorflow)"""
import tensorflow as tf


def lenet5(x, y):
    """wiw wiw"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=(5, 5),
                             kernel_initializer=kernel,
                             padding="SAME", activation='relu')

    pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2))

    conv3 = tf.layers.conv2d(pool1, filters=16, kernel_size=(5, 5),
                             kernel_initializer=kernel,
                             padding="VALID", activation='relu')

    pool2 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2))

    flat = tf.contrib.layers.flatten(pool2)

    layer1 = tf.layers.dense(flat, units=120,
                             kernel_initializer=kernel, activation='relu')

    layer2 = tf.layers.dense(layer1, units=84,
                             kernel_initializer=kernel, activation='relu')

    layer3 = tf.layers.dense(layer2, units=10,
                             kernel_initializer=kernel)

    y_pred = tf.nn.softmax(layer3)

    loss = tf.losses.softmax_cross_entropy(y, layer3)

    optimizer = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(layer3, axis=1),
                                  tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

    return y_pred, optimizer, loss, accuracy
