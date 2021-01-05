#!/usr/bin/env python3
"""class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """class DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__weights['W1'] = np.random.normal(size=(layers[0], nx)) \
            * np.sqrt(2/nx)
        self.__weights['b1'] = np.zeros((layers[0], 1))
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i != 0:
                self.__weights['W' + str(i + 1)
                               ] = np.random.normal(size=(layers[i],
                                                          layers[i-1])) \
                     * np.sqrt(2/(layers[i-1]))
                self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            z = np.matmul(self.__weights['W' + str(i + 1)],
                          self.__cache['A' + str(i)
                                       ]) + self.__weights['b' + str(i + 1)]
            self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-z))
        return self.cache['A' + str(i+1)], self.__cache

    def cost(self, Y, A):
        """calcul cost"""
        s = (-Y) * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
        return np.sum(s)/Y.shape[1]

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A2 = self.forward_prop(X)[0]
        A3 = np.round(A2).astype(int)
        return A3, self.cost(Y, A2)
