#!/usr/bin/env python3
"""class NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """class NeuralNetwork"""
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """forward propagation"""
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """calcul cost"""
        s = (-Y) * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
        return np.sum(s)/Y.shape[1]

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A2 = self.forward_prop(X)[1]
        A3 = np.round(A2).astype(int)
        return A3, self.cost(Y, A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = len(X[0])
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) * 1 / m
        db2 = np.sum(dz2, axis=1, keepdims=True) * 1 / m
        dA1 = A1 * (1 - A1)
        dz1 = np.matmul(self.__W2.T, dz2) * dA1
        dw1 = np.matmul(dz1, X.T) * 1 / m
        db1 = np.sum(dz1, axis=1, keepdims=True) * 1 / m
        self.__W2 = self.__W2 - dw2 * alpha
        self.__b2 = self.__b2 - db2 * alpha
        self.__W1 = self.__W1 - dw1 * alpha
        self.__b1 = self.__b1 - db1 * alpha
