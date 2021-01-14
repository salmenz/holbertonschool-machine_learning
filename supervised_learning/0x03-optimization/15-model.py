#!/usr/bin/env python3
"""builds, trains, and saves a neural network model in tensorflow using Adam
optimization, mini-batch gradient descent, learning rate decay, and batch
normalization"""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """ tf.placeholder"""
    x = tf.placeholder(tf.float32, [None, nx], name="x")
    y = tf.placeholder(tf.float32, [None, classes], name="y")
    return x, y


def create_layer(prev, n, activation):
    """create layer with tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    la = tf.layers.Dense(units=n, activation=activation, name="layer",
                         kernel_initializer=kernel)
    return la(prev)


def forward_prop(x, layer_sizes=[], activations=[], epsilon=1e-8):
    """the forward propagation for the neural network"""
    for i in range(len(layer_sizes)):
        if i == len(layer_sizes) - 1:
            x = create_layer(x, layer_sizes[i], activations[i])
        else:
            x = create_batch_norm_layer(x, layer_sizes[i], activations[i],
                                        epsilon)
    return x


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    pred = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    acc = tf.reduce_mean(tf.cast(pred, "float32"))
    return acc


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates a learning rate decay operation in tensorflow
     using inverse time decay"""
    return tf.train.inverse_time_decay(alpha, global_step,
                                       decay_step, decay_rate, staircase=True)


def create_batch_norm_layer(prev, n, activation, epsilon):
    """creates batch normalization layer for a neural network in tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, name="layer",
                            kernel_initializer=kernel)
    mean, variance = tf.nn.moments(layer(prev), [0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    Bnorm = tf.nn.batch_normalization(layer(prev), mean, variance, beta,
                                      gamma, epsilon)
    return activation(Bnorm)


def shuffle_data(X, Y):
    """shuffles the data points in
     two matrices the same way:"""
    perm = np.random.permutation(len(X))
    x = X[perm]
    y = Y[perm]
    return x, y


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """that creates the training operation for a neural network in tensorflow
     using the Adam optimization algorithm"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2,
                                  epsilon).minimize(loss)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations, epsilon)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    nbr_batches = X_train.shape[0]//batch_size
    if X_train.shape[0] % batch_size != 0:
        nbr_batches += 1
    global_step = tf.Variable(0, trainable=False)
    step = tf.assign_add(global_step, 1, name="step")
    alpha = learning_rate_decay(alpha, decay_rate, global_step, nbr_batches)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epochs+1):
            X, Y = shuffle_data(X_train, Y_train)
            tacc, tloss = sess.run((accuracy, loss), {x: X_train, y: Y_train})
            vacc, vloss = sess.run((accuracy, loss), {x: X_valid, y: Y_valid})
            print("After {} epochs:".format(ep))
            print("\tTraining Cost: {}".format(tloss))
            print("\tTraining Accuracy: {}".format(tacc))
            print("\tValidation Cost: {}".format(vloss))
            print("\tValidation Accuracy: {}".format(vacc))
            if ep != epochs:
                for i in range(batch_size, X.shape[0]+batch_size, batch_size):
                    xdat, ydat = X[i-batch_size:i], Y[i-batch_size:i]
                    sess.run((train_op, step), {x: xdat, y: ydat})
                    dl, da = sess.run((loss, accuracy), {x: xdat, y: ydat})
                    if i % (batch_size*100) == 0 and i != 0:
                        print("\tStep {}:".format(i//batch_size))
                        print("\t\tCost: {}".format(dl))
                        print("\t\tAccuracy: {}".format(da))
        return saver.save(sess, save_path)
