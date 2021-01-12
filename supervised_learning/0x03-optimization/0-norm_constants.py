#!/usr/bin/env python3
"""calculates the normalization (standardization)"""
import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization)"""
    m, nx = X.shape
    mean = np.sum(X, axis=0)/m
    s = np.sum((X - mean) ** 2, axis=0) * 1/m
    return mean, np.sqrt(s)
