#!/usr/bin/env python3
"""determines the steady state probabilities of a regular markov chain"""
import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain"""

    if np.any(P == 0):
        return None
    s = np.zeros(1, P.shape[0])
    s[0, 0] = 1
    while not np.array_equal(np.matmul(s, P), s):
        s = np.matmul(s, P)
    return s
