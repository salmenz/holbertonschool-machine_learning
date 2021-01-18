#!/usr/bin/env python3
"""calculates the sensitivity for each class in a precision matrix"""
import numpy as np


def precision(confusion):
    """calculates the sensitivity for each class in a precision matrix"""
    t = np.zeros(confusion.shape[0])
    c = confusion.T
    i = 0
    for cl in c:
        t[i] = cl[i] / cl.sum()
        i += 1
    return t
