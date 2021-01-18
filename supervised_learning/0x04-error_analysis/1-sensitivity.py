#!/usr/bin/env python3
"""calculates the sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix"""
    t = np.zeros(confusion.shape[0])
    i = 0
    for cl in confusion:
        t[i] = cl[i] / cl.sum()
        i += 1
    return t
