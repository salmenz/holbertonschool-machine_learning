#!/usr/bin/env python3
"""calculates the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """calculates the probability density
    function of a Gaussian distribution"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[1] != S.shape[0]:
        return None
    m1 = X - m
    inv_S = np.linalg.inv(S)
    mul1 = np.matmul(m1, inv_S)
    var1 = np.exp(np.matmul(mul1, m1.T).diagonal() * -0.5)
    var2 = (((2 * np.pi) ** X.shape[1]) * np.linalg.det(S)) ** (0.5)
    pdf = var1/var2
    return np.where(pdf < 1e-300, 1e-300, pdf)
