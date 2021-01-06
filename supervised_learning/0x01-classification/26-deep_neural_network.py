#!/usr/bin/env python3
"""class DeepNeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        weights = self.__weights.copy()
        dz = self.cache['A' + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            dwn = np.matmul(dz, cache['A' + str(i - 1)].T) * 1 / m
            dbn = np.sum(dz, axis=1, keepdims=True) * 1 / m
            self.__weights['W' + str(i)] = self.__weights['W' + str(i)
                                                          ] - dwn * alpha
            self.__weights['b' + str(i)] = self.__weights['b' + str(i)
                                                          ] - dbn * alpha
            dA = self.cache['A' + str(i - 1)] * (1 - self.cache['A'+str(i-1)])
            dz = np.matmul(weights['W' + str(i)].T, dz) * dA

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """"Trains the deep neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        x = []
        y = []
        for i in range(iterations+1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if verbose:
                A = self.cost(Y, self.__cache['A'+str(self.L)])
                print(f"Cost after {i} iterations: {A}")
                if i == 0 or i % 100 == 0:
                    x.append(i)
                    y.append(A)
        if graph:
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)[0], y[len(y)-1]

    def save(self, filename):
        """save pickle file"""
        if not filename:
            return None
        if not(filename.endswith(".pkl")):
            filename = filename + ".pkl"
        with open(filename, 'wb') as fileObject:
            return pickle.dump(self, fileObject)

    @staticmethod
    def load(filename):
        """load pickle file"""
        try:
            with open(filename, 'rb') as fileObject:
                r = pickle.load(fileObject)
            return r
        except FileNotFoundError:
            return None
