#!/usr/bin/env python3
"""calculates the total intra-cluster variance for a data set"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    return np.sum(np.square(np.min(distances, axis=0)))
