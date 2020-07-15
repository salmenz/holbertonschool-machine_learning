#!/usr/bin/env python3
'''Gettin’ Cozy'''


def cat_matrices2D(mat1, mat2, axis=0):
    m = []
    """that concatenates two matrices along a specific axis"""
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        m = mat1+mat2
        return m
    elif axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            l1 = mat1[i]+mat2[i]
            m.append(l1)
        return mat1
