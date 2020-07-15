#!/usr/bin/env python3
'''Ridin’ Bareback'''
import numpy as np


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
    for l in mat1:
        if len(l) != len(mat1[0]):
            return None
    if len(mat1[0]) != len(mat2):
        return None
    a = np.array(mat1)
    b = np.array(mat2)
    return np.matmul(a, b)
