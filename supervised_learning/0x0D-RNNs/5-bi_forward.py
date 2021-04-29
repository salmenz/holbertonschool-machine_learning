#!/usr/bin/env python3
""" bidirectional Cell RNN """
import numpy as np


class BidirectionalCell():
    """class bidirectionalcell"""
    def __init__(self, i, h, o):
        """class constructor """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """represents a bidirectional cell of an RNN"""
        n = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(n, self.Whf) + self.bhf)
        return h_next
