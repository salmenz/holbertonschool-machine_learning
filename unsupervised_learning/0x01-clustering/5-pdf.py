#!/usr/bin/env python3
"""calculates the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """calculates the probability density
    function of a Gaussian distribution"""
    m1 = X - m
    inv_S = np.linalg.inv(S)
    var1 = np.exp((np.matmul(np.matmul(m1, inv_S), m1.T)) * -0.5)
    var2 = (((2 * np.pi) ** X.shape[1]) * np.linalg.det(S)) ** (0.5)
    return var1/var2
