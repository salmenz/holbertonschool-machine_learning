#!/usr/bin/env python3
"""performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    X = X - np.mean(X, axis=0)
    S = np.linalg.svd(X)[2][:ndim, :].T
    return np.matmul(X, S)
