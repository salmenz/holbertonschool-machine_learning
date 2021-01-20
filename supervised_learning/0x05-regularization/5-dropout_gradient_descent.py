#!/usr/bin/env python3
"""updates the weights of a neural network with Dropout
regularization using gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network with Dropout
    regularization using gradient descent"""
    m = Y.shape[1]
    weights = weights.copy()
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        dwn = np.matmul(dz, cache['A' + str(i - 1)].T) * 1 / m
        dbn = np.sum(dz, axis=1, keepdims=True) * 1 / m
        weights['W' + str(i)] = weights['W' + str(i)] - dwn * alpha
        weights['b' + str(i)] = weights['b' + str(i)] - dbn * alpha
        dA = cache['A' + str(i - 1)] * (1 - cache['A'+str(i-1)])
        if i > 1:
            dz = np.matmul(weights['W' + str(i)].T, dz) * dA * \
                 cache['D' + str(i-1)] / keep_prob
