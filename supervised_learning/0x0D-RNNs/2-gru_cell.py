#!/usr/bin/env python3
import numpy as np
"""class GRUCell"""


class GRUCell:
    """class GRUCell"""
    def __init__(self, i, h, o):
        """ class constructor"""
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid function"""
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def softmax(self, x):
        """ softmax function"""
        exp_scores = np.exp(x)
        return exp_scores/np.sum((exp_scores), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ forward propagation for one time step RNN GRU"""
        U = np.concatenate((h_prev, x_t), axis=1)
        sig = self.sigmoid(np.dot(U, self.Wr) + self.br)
        sigm = self.sigmoid(np.dot(U, self.Wz) + self.bz)

        V = np.concatenate(((sig * h_prev), x_t), axis=1)
        tan = np.tanh(np.dot(V, self.Wh) + self.bh)
        h_next = (1 - sigm) * h_prev + sigm * tan
        y = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y)
        return h_next, y
