#!/usr/bin/env python3
"""calculates the F1 score for each class in a precision matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score for each class in a precision matrix"""
    return 2 * ((precision(confusion) * sensitivity(confusion)) /
                (precision(confusion) + sensitivity(confusion)))
