#!/usr/bin/env python3
"""Class GaussianProcess"""
import numpy as np


class GaussianProcess():
    """represents a noiseless 1D Gaussian process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix between two matrices"""
        s = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)
        dt = s + (- 2 * np.dot(X1, X2.T))
        K = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * dt)
        return K

    def predict(self, X_s):
        """predicts the mean and standard deviation
        of points in a Gaussian process"""
        kernel = self.kernel(self.X, X_s)
        s_ker = self.kernel(X_s, X_s)
        inv_ker = np.linalg.inv(self.K)
        s = kernel.T.dot(inv_ker).dot(self.Y)
        mul = np.reshape(s, -1)
        cs = s_ker - kernel.T.dot(inv_ker).dot(kernel)
        sig = np.diagonal(cs)
        return mul, sig
