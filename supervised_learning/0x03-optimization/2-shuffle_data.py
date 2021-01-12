#!/usr/bin/env python3
"""shuffles the data points in
 two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in
     two matrices the same way:"""
    perm = np.random.permutation(len(X))
    x = X[perm]
    y = Y[perm]
    return x, y
