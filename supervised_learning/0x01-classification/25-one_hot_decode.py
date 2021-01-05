#!/usr/bin/env python3
"""one hot decode"""
import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels:"""
    if len(one_hot.shape) != 2 or type(one_hot) is not np.ndarray:
        return None
    return np.argmax(one_hot, axis=0)        

