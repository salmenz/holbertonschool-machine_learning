#!/usr/bin/env python3
'''racing The Elements'''
import numpy as np


def np_elementwise(mat1, mat2):
    """ performs element-wise addition, subtraction, multiplication, and division"""
    a = np.array(mat1)
    b = np.array(mat2)
    return (np.add(a, b), np.subtract(a, b), np.multiply(a, b), np.true_divide(a, b))
    