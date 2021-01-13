#!/usr/bin/env python3
"""normalizes an unactivated output of a
 neural network using batch normalization:"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a
    neural network using batch normalization"""
    mean = np.mean(Z, axis=0)
    variance = np.var((Z - mean), axis=0)
    Znorm = (Z - mean) / (np.sqrt(variance + epsilon))
    return gamma * Znorm + beta
