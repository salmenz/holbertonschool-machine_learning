#!/usr/bin/env python3
"""that performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """that performs PCA on a dataset"""
    V = np.linalg.svd(X)[1]
    V = V / sum(V)
    S = 0
    n = 0
    while S < var:
        S += V[n]
        n += 1
    W = np.linalg.svd(X)[2]
    return W[:n, :].T
