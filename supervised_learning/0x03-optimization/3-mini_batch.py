#!/usr/bin/env python3
"""rains a loaded neural network model using mini-batch gradient descent"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """rains a loaded neural network model using mini-batch gradient descent"""
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph('{}.meta'.format(load_path))
        saver.restore(sess, '{}'.format(load_path))
        train_op = tf.get_collection('train_op', scope=None)[0]
        loss = tf.get_collection('loss', scope=None)[0]
        accuracy = tf.get_collection('accuracy', scope=None)[0]
        x = tf.get_collection('x', scope=None)[0]
        y = tf.get_collection('y', scope=None)[0]
        print(X_train.shape[0])
        X_train, Y_train = shuffle_data(X_train, Y_train)
        for ep in range(epochs+1):
            tacc = sess.run(accuracy, {x: X_train, y: Y_train})
            tloss = sess.run(loss, {x: X_train, y: Y_train})
            vacc = sess.run(accuracy, {x: X_valid, y: Y_valid})
            vloss = sess.run(loss, {x: X_valid, y: Y_valid})
            print("After {} epochs:".format(ep))
            print("\tTraining Cost: {}".format(tloss))
            print("\tTraining Accuracy: {}".format(tacc))
            print("\tValidation Cost: {}".format(vloss))
            print("\tValidation Accuracy: {}".format(vacc))
            if ep != epochs:
                for i in range(batch_size, X_train.shape[0]+ batch_size, batch_size):
                    xdat = X_train[i-batch_size:i]
                    ydat = Y_train[i-batch_size:i]
                    sess.run(train_op, {x: xdat, y: ydat})
                    if i % (batch_size*100) == 0 and i != 0:
                        dl, da = sess.run((loss, accuracy), {x: xdat, y: ydat})
                        print("\tStep {}:".format(i/batch_size))
                        print("\t\tCost: {}".format(dl))
                        print("\t\tAccuracy: {}".format(da))
        return saver.save(sess, save_path)
