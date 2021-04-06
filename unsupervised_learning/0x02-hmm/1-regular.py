#!/usr/bin/env python3
"""determines the steady state probabilities of a regular markov chain"""
import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain"""
    if type(P) != np.ndarray or len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if not np.array_equal(np.sum(P, axis=1), np.ones(P.shape[0])):
        return None
    if np.max(P) > 1 or np.min(P) < 0:
        return None
    s = np.zeros(P.shape[0])
    s[0] = 1
    while not np.array_equal(np.matmul(s, P), s):
        s = np.matmul(s, P)
    if np.any(s == 0):
        return None
    return s
