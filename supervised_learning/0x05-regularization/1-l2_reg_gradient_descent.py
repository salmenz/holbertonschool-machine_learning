#!/usr/bin/env python3
"""updates the weights and biases of a neural network
using gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network using gradient
    descent with L2 regularization"""
    m = Y.shape[1]
    weights_cpy = weights.copy()
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        dwn = (np.matmul(dz, cache['A' + str(i - 1)].T) * 1 / m) + \
            (weights['W' + str(i)] * lambtha/m)
        dbn = np.sum(dz, axis=1, keepdims=True) * 1 / m
        weights['W' + str(i)] = weights['W' + str(i)] - dwn * alpha
        weights['b' + str(i)] = weights['b' + str(i)] - dbn * alpha
        dA = cache['A' + str(i - 1)] * (1 - cache['A'+str(i-1)])
        dz = np.matmul(weights_cpy['W' + str(i)].T, dz) * dA
