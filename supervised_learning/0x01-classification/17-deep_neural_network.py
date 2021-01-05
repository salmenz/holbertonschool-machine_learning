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
