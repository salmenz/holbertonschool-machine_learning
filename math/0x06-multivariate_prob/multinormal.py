#!/usr/bin/env python3
"""Class MultiNormal"""
import numpy as np


class MultiNormal:
    """Class Multinormal"""
    def __init__(self, data):
        """represents a Multivariate Normal distribution"""
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = data.mean(axis=1, keepdims=True)
        self.cov = np.matmul(data - self.mean, np.transpose(data - self.mean))\
            / (data.shape[1] - 1)
