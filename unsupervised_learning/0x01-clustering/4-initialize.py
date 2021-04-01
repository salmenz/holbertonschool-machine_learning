#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans
"""initializes variables for a Gaussian Mixture Model"""


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    d = X.shape[1]
    pi = np.full(k, 1/k)
    S = np.full((k, d, d), np.identity(2))
    return pi, kmeans(X, k)[0], S
