#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X
    for i in range(L):
        z = np.matmul(weights['W' + str(i + 1)], cache['A' + str(i)]) + \
            weights['b' + str(i + 1)]
        if i != L - 1:
            cache['A' + str(i + 1)] = np.tanh(z)
            cache['D'+str(i+1)] = np.random.rand(cache['A'+str(i+1)].shape[0],
                                                 cache['A' + str(i + 1)]
                                                 .shape[1])
            cache['D'+str(i+1)] = cache['D'+str(i+1)] < keep_prob
            cache['D'+str(i+1)] = np.multiply(cache['D'+str(i+1)], 1)
            cache['A' + str(i+1)] *= cache['D'+str(i+1)]
            cache['A' + str(i+1)] /= keep_prob
        else:
            t = np.exp(z)
            cache['A' + str(i + 1)] = t / np.sum(t, axis=0)
    return cache
