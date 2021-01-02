#!/usr/bin/env python3
"""class neuron"""
import numpy as np


class Neuron:
    """class neuron"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """forward propagation"""
        mul = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-mul))
        return self.__A

    def cost(self, Y, A):
        """calcul cost"""
        s = (-Y) * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
        return np.sum(s)/Y.shape[1]

    def evaluate(self, X, Y):
        """evaluate the neuron's predection"""
        ev = np.round(self.forward_prop(X)).astype(int)
        return ev, self.cost(Y, self.forward_prop(X))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gardient_descent"""
        self.__W = self.__W - np.sum((A - Y)*X, axis=1) * alpha * 1/len(X[0])
        self.__b = self.__b - np.sum(A - Y)/Y.shape[1] * alpha
