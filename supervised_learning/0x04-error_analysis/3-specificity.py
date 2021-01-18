#!/usr/bin/env python3
"""calculates the sensitivity for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """calculates the sensitivity for each class in a confusion matrix"""
    t = np.zeros(confusion.shape[0])
    i = 0
    confusion = confusion.T
    for cl in confusion:
        TN = 0
        for k in range(confusion.shape[0]):
            for j in range(confusion.shape[0]):
                if j != i and k != i:
                    TN += confusion[k][j]
        t[i] = TN / (TN + cl.sum() - cl[i])
        i += 1
    return t
