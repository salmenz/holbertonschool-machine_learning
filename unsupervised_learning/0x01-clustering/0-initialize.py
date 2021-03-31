#!/usr/bin/env python3
"""initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    d = X.shape[1]
    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))
