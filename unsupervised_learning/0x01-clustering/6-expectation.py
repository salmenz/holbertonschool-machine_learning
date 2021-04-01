#!/usr/bin/env python3
"""calculates the expectation step in the EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if not np.isclose(pi.sum(), 1):
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    k = pi.shape[0]
    if (k, X.shape[1]) != m.shape or (k, X.shape[1], X.shape[1]) != S.shape:
        return None, None
    g_ar = []
    for i in range(k):
        P = pdf(X, m[i], S[i]) * pi[i]
        g_ar.append(P)
    g_ar = np.array(g_ar)
    lh = np.log(g_ar.sum(axis=0))
    sum_lh = np.sum(lh)
    g_ar /= g_ar.sum(axis=0)
    return g_ar, sum_lh
