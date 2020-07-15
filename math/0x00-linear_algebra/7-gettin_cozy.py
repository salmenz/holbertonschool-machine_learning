#!/usr/bin/env python3
'''Gettin’ Cozy'''


def cat_matrices2D(mat1, mat2, axis=0):
    """that concatenates two matrices along a specific axis"""
    m1 = []
    m2 = []
    if axis == 0:
        return mat1+mat2
    else:
        for i in range(len(mat1)):
            for j in range(axis):
                mat1[i].append(mat2[i][j])
        return mat1
