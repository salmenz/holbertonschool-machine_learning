#!/usr/bin/env python3
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    try:
        b = np.zeros((Y.size, classes))
        b[np.arange(Y.size), Y] = 1
        return b.T
    except Exception:
        return None
