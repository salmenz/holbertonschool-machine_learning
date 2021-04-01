#!/usr/bin/env python3
"""calculates the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """calculates the maximization step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
