#!/usr/bin/env python3
"""bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """performs forward propagation for a bidirectional RNN"""
    t, m, i = X.shape
    HH = []
    Hb = []
    h_prev = h_0
    Hs = h_t
    for i in range(t):
        h_prev = bi_cell.forward(h_prev, X[i])
        Hs = bi_cell.backward(Hs, X[t-i-1])
        HH.append(h_prev)
        Hb.append(Hs)

    Hb = [x for x in reversed(Hb)]
    H = np.concatenate((np.array(HH), np.array(Hb)), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
