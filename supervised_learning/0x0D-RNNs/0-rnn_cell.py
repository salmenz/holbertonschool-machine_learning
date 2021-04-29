#!/usr/bin/env python3
"""class RNNCell"""
import numpy as np


def softmax(x):
    """softmax function"""
    exp_scores = np.exp(x)
    return exp_scores/np.sum((exp_scores), axis=1, keepdims=True)


class RNNCell:
    """class RNNCell"""
    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward"""
        n = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(n, self.Wh) + self.bh)
        y = softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
