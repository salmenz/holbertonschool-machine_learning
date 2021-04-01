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
    g = []
    for i in range(k):
        P = pdf(X, m[i], S[i]) * pi[i]
        g.append(P)
    g = np.array(g)
    lh = np.log(g.sum(axis=0))
    tot_likel = np.sum(lh)
    post = g / g.sum(axis=0)
    return post, tot_likel
