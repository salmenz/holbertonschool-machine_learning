#!/usr/bin/env python3
"""evaluates the output of a neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network"""
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph('{}.meta'.format(save_path))
        saver.restore(sess, '{}'.format(save_path))
        y_pred = tf.get_collection('y_pred', scope=None)[0]
        loss = tf.get_collection('loss', scope=None)[0]
        acc = tf.get_collection('accuracy', scope=None)[0]
        x = tf.get_collection('x', scope=None)[0]
        y = tf.get_collection('y', scope=None)[0]

        y_pred = sess.run(y_pred, {x: X, y: Y})
        acc = sess.run(acc, {x: X, y: Y})
        loss = sess.run(loss, {x: X, y: Y})

    return(y_pred, acc, loss)
