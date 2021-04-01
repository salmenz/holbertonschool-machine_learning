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
    n, d = X.shape
    if d != m.shape[0] or (d, d) != S.shape:
        return None
    d = X.shape[1]
    K = 1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(S)))
    b = np.dot((X - m), np.linalg.inv(S))
    bb = -0.5 * np.dot(b, (X - m).T).diagonal()
    P = K * np.exp(bb)
    P = np.where(P >= 1e-300, P, 1e-300)
    return P
