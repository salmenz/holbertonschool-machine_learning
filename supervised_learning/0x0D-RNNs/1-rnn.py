#!/usr/bin/env python3
import numpy as np
"""performs forward propagation for a simple RNN"""


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN"""
    t = X.shape[0]
    H = []
    H.append(h_0)
    Y = []
    for i in range(t):
        h_next, y = rnn_cell.forward(H[i], X[i])
        H.append(h_next)
        Y.append(y)
    return np.array(H), np.array(Y)
