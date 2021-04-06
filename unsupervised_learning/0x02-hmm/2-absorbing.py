#!/usr/bin/env python3
"""determines if a markov chain is absorbing"""
import numpy as np


def getkeys(keys, i, P):
    """get keys"""
    X = P.T[i]
    for j in range(P.shape[0]):
        if X[j] > 0:
            keys.append(j)
    return keys


def absorbing(P):
    """determines if a markov chain is absorbing"""
    if type(P) != np.ndarray or len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False
    if not np.array_equal(np.sum(P, axis=1), np.ones(P.shape[0])):
        return False
    if np.max(P) > 1 or np.min(P) < 0:
        return False
    x = np.diagonal(P)

    if not np.any(x == 1):
        return False
    if np.all(x == 1):
        return True
    keys = []
    for i in range(len(x)):
        if x[i] == 1:
            keys.append(i)

    for i in range(P.shape[0]):
        if i in keys:
            keys = getkeys(keys, i, P)

    for i in range(P.shape[0]):
        if i in keys:
            keys = getkeys(keys, i, P)
    return len(set(keys)) == P.shape[0]
