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
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        self.weights['W1'] = np.random.normal(size=(layers[0], nx)) \
            * np.sqrt(2/nx)
        self.weights['b1'] = np.zeros((layers[0], 1))
        for i in range(1, self.L):
            self.weights['W{i+1}'] = np.random.randn(layers[i], layers[i-1]) \
                     * np.sqrt(2/(layers[i] + layers[i-1]))
            self.weights['b{i+1}'] = np.zeros((layers[i], 1))
