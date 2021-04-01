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
    n = X.shape[0]
    k = S.shape[0]
    g = np.zeros((k, n))
    for cluster in range(k):
        prob = pdf(X, m[cluster], S[cluster])
        prior = pi[cluster]
        g[cluster] = prior * prob
    total = np.sum(g, axis=0, keepdims=True)
    post = g / total
    tot_likel = np.sum(np.log(total))
    return post, tot_likel
