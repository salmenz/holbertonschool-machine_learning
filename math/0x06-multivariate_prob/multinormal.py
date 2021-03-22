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

    def pdf(self, x):
        """calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        if x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        n = x.shape[0]
        m = self.mean
        c = self.cov
        den = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(c))
        icov = np.linalg.inv(c)
        expo = (-0.5 * np.matmul(np.matmul((x - m).T, icov), x - self.mean))
        pdf = (1 / den) * np.exp(expo[0][0])
        return pdf
