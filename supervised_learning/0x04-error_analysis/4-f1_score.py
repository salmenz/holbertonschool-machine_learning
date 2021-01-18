#!/usr/bin/env python3
"""calculates the F1 score for each class in a precision matrix"""
import numpy as np


def f1_score(confusion):
    """calculates the F1 score for each class in a precision matrix"""
    t = np.zeros(confusion.shape[0])
    t1 = np.zeros(confusion.shape[0])
    c = confusion.T
    i = 0
    for cl, cl1 in zip(c, confusion):
        t[i] = cl[i] / cl.sum()
        t1[i] = cl1[i] / cl1.sum()
        i += 1
    return 2 * t * t1 / (t1 + t)