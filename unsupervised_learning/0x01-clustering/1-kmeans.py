#!/usr/bin/env python3
"""performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    C = np.random.uniform(min, max, (k, X.shape[1]))
    for _ in range(iterations):
        C_copy = np.copy(C)
        # calculate the distance between all points and all centroids
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        # append every points to the closest cluster
        clss = distances.argmin(axis=0)
        for j in range(k):
            if (X[clss == j].size == 0):
                C[j] = np.random.uniform(min, max, size=(1, X.shape[1]))
            else:
                C[j] = (X[clss == j].mean(axis=0))
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        clss = distances.argmin(axis=0)
        if (C_copy == C).all():
            return C, clss
    return C, clss
