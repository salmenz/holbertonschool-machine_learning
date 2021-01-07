#!/usr/bin/env python3
"""
 DeepNeuralNetwork that defines a deep
 neural network performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    class DeepNeuralNetwork
    nx is the number of input features
    layers is a list representing the number
    of nodes in each layer of the network
    L: The number of layers in the neural network
    """

    def __init__(self, nx, layers, activation='sig'):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx <= 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights['b' + str(i + 1)] = np.zeros((layers[0], 1))
                self.__weights['W' + str(i + 1)
                               ] = np.random.normal(size=(layers[i], nx)
                                                    ) * np.sqrt(2/nx)

            else:
                self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
                self.__weights[
                    'W' + str(i + 1)
                ] = np.random.normal(size=(layers[i], layers[i-1])
                                     )*np.sqrt(
                    2/layers[i-1])

    @property
    def L(self):
        """The number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """A dictionary to hold all intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """A dictionary to hold all weights and biased of the network"""
        return self.__weights

    @property
    def activation(self):
        """activation function"""
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        X should be saved to the cache dictionary using the key A0
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            w = self.__weights["W"+str(i+1)]
            b = self.__weights["b" + str(i+1)]
            z = np.matmul(w, self.__cache["A"+str(i)]) + b
            if(i != self.__L - 1):
                if self.__activation == 'sig':
                    Sigmoid_a = 1 / (1 + np.exp(-z))
                    self.__cache["A"+str(i+1)] = Sigmoid_a
                elif self.__activation == 'tanh':
                    tanh = (2 / (1 + np.exp(-2 * z))) - 1
                    self.__cache["A"+str(i+1)] = tanh

            else:
                """ softmax for the output layer
                Softmax function returns probabilities sum to 1
                OR
                t =np.exp(z)
                sumT = np.sum(t, axis=0)
                softmax = t/sumT
                """
                t = np.exp(z - np.max(z))
                softmax = t / t.sum(axis=0)
                self.__cache["A"+str(i+1)] = softmax

        return self.__cache["A"+str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        cost for softmax
        Y is now a one-hot numpy.ndarray of shape (classes, m)
        classes is the maximum number of classes
        m is the number of examples
        """
        classes, m = Y.shape
        loss = - (Y * np.log(A))
        sumloss = np.sum(loss)
        cost = (1 / m) * sumloss
        return cost

    def evaluate(self, X, Y):
        """ evaluate The activated output
        Softmax function returns probabilities sum to 1
        """
        softmax, cache = self.forward_prop(X)
        pred_evalute = np.where(softmax == np.amax(softmax, axis=0), 1, 0)
        cost = self.cost(Y, softmax)
        return pred_evalute, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """gradient_descent
        backpropagation
        """
        weights = self.__weights.copy()
        nx, m = Y.shape
        dz = cache["A"+str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_i = "A"+str(i-1)
            wi = "W"+str(i)
            bi = "b"+str(i)
            dw = (1/m) * np.matmul(dz, self.__cache["A"+str(i-1)].T)
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            self.__weights[wi] = self.__weights[wi] - (dw * alpha)
            self.__weights[bi] = self.__weights[bi] - (db * alpha)
            if self.__activation == 'sig':
                dz = np.matmul(weights[wi].T, dz
                               ) * (cache[A_i] * (1 - cache[A_i]))
            elif self.__activation == 'tanh':
                dz = np.matmul(weights[wi].T, dz
                               ) * (1 - np.power(cache[A_i], 2))
        return self.__weights

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Returns the evaluation of the training data after
        iterations of training have occurred
        used:
        -forward_prop
        -gradient_descent
        - evaluate
        """
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
        xValue = []
        yValue = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)
            if verbose:
                if (i < 1 or i % step == 0):
                    print("Cost after {} iterations: {}".format(i, cost))
                    xValue.append(i+step)
                    yValue.append(cost)
        if graph is True:
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iteration')
            plt.plot(xValue, yValue)
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ save file .pkl"""
        if not filename:
            return None
        if not(filename.endswith(".pkl")):
            filename = filename + ".pkl"
        with open(filename, 'wb') as fileObject:
            return pickle.dump(self, fileObject)

    @staticmethod
    def load(filename):
        """ load file .pkl"""
        try:
            with open(filename, 'rb') as fileObject:
                res = pickle.load(fileObject)
            return res
        except FileNotFoundError:
            return None
