#!/usr/bin/env python3
"""GRUCell RNN"""
import numpy as np


class LSTMCell():
    """class LSTMCell"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """ Sigmoid fn """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def softmax(self, x):
        """softmax function"""
        exp_scores = np.exp(x)
        return exp_scores/np.sum((exp_scores), axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """fn that performs forward propagation for one time"""
        co = np.concatenate((h_prev, x_t), axis=1)
        U = self.sigmoid(np.matmul(co, self.Wf) + self.bf)
        V = self.sigmoid(np.matmul(co, self.Wu) + self.bu)
        C = np.tanh(np.matmul(co, self.Wc) + self.bc)
        fin_C = U * c_prev + V * C
        s3_O = self.sigmoid(np.matmul(co, self.Wo) + self.bo)
        h_next = s3_O * np.tanh(fin_C)
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, fin_C, y
