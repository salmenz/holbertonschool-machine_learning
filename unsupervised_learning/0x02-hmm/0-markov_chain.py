#!/usr/bin/env python3
"""determines the probability of a markov chain being in a particular
state after a specified number of iterations"""
import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain being in a particular
    state after a specified number of iterations"""
    if type(t) != int or t < 1:
        return None
    if type(P) != np.ndarray or len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if type(s) != np.ndarray or s.shape[0] != 1 or P.shape[0] != s.shape[1]:
        return None
    for _ in range(t):
        s = np.matmul(s, P)
    return s
