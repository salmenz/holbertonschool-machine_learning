#!/usr/bin/env python3
'''Across The Planes'''


def add_matrices2D(mat1, mat2):
    """two matrices element-wise"""
    if len(mat2) != len(mat1):
        return None
    m = []
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
        l1 = []
        for j in range(len(mat1[i])):
            l1.append(mat2[i][j]+mat1[i][j])
        m.append(l1)
    return(m)
