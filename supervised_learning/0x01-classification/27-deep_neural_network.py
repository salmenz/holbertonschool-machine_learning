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
            if i != self.__L - 1:
                self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(- z))
            else:
                t = np.exp(z)
                self.__cache['A' + str(i + 1)] = t / np.sum(t, axis=0)
        return self.cache['A' + str(i+1)], self.__cache

    def cost(self, Y, A):
        """calcul cost"""
        m = Y.shape[1]
        s = - Y * np.log(A)
        return 1/m*np.sum(s)

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A2 = self.forward_prop(X)[0]
        A3 = np.zeros(A2.shape)
        A3[np.argmax(A2, axis=0)] = 1
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
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        x = []
        y = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)
            if verbose:
                if (i < 1 or i % step == 0):
                    print("Cost after {} iterations: {}".format(i, cost))
                    x.append(i+step)
                    y.append(cost)
        if graph is True:
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iteration')
            plt.plot(x, y)
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """save pickle file"""
        if not filename:
            return None
        if not(filename.endswith(".pkl")):
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            return pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """load pickle file"""
        try:
            with open(filename, 'rb') as f:
                r = pickle.load(f)
            return r
        except FileNotFoundError:
            return None
