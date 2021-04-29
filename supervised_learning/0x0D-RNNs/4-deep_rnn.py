#!/usr/bin/env python3
"""deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """FN performs forward propagation for a deep RNN"""
    t = X.shape[0]
    H = []
    H.append(h_0)
    Y = []
    for x in range(t):
        k = 0
        hh = []
        h = X[x]
        for k, rnn in enumerate(rnn_cells):
            h, y = rnn.forward(H[x][k], h)
            hh.append(h)
        H.append(hh)
        Y.append(y)
    return np.array(H), np.array(Y)
