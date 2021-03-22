#!/usr/bin/env python3
"""calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    variance = np.sqrt(np.diag(C))
    outer = np.outer(variance, variance)
    C = C / outer
    return C
