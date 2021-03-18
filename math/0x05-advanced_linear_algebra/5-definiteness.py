#!/usr/bin/env python3
"""calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) == 1:
        return None

    if not np.array_equal(matrix.T, matrix):
        return None

    if np.all(np.linalg.eigvals(matrix) > 0):
        return("Positive definite")
    elif np.all(np.linalg.eigvals(matrix) < 0):
        return("Negative definite")
    elif np.all(np.linalg.eigvals(matrix) >= 0):
        return("Positive semi-definite")
    elif np.all(np.linalg.eigvals(matrix) <= 0):
        return("Negative semi-definite")
    else:
        return("Indefinite")
