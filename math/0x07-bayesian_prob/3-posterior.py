#!/usr/bin/env python3
"""calculates the marginal probability of obtaining data"""

import numpy as np


def posterior(x, n, P, Pr):
    """calculates the marginal probability of obtaining data"""
    if not isinstance(n, int) or (n <= 0):
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or (x < 0):
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError('All values in P must be in the range [0, 1]')
    if not isinstance(Pr, np.ndarray) or (P.shape != Pr.shape):
        err = 'Pr must be a numpy.ndarray with the same shape as P'
        raise TypeError(err)
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    E = np.sum(Pr)
    if not np.isclose(E, 1):
        raise ValueError('Pr must sum to 1')

    fact = (np.math.factorial(n)) / \
        (np.math.factorial(x) * np.math.factorial(n - x))
    fact *= (np.power(P, x)) * (np.power((1 - P), (n - x)))
    inter = fact * Pr
    marg = np.sum(inter)
    return inter / marg
