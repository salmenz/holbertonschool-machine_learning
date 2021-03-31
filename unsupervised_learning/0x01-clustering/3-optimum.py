#!/usr/bin/env python3
"""tests for the optimum number of clusters by variance"""
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if kmin > X.shape[0] or kmax >= X.shape[0]:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    results = []
    d_vars = []
    c_min = kmeans(X, kmin)[0]
    var_min = variance(X, c_min)
    for i in range(kmin, kmax + 1):
        C, clss = kmeans(X, i, iterations)
        results.append((C, clss))
        d_vars.append(var_min - variance(X, C))
    return results, d_vars
