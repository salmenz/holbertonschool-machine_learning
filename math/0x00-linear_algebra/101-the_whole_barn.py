#!/usr/bin/env python3
'''The Whole Barn'''


def mat_shape(matrix):
    """matrix shape"""
    shape = []
    if type(matrix) != list:
        pass
    else:
        shape.append(len(matrix))
        shape += mat_shape(matrix[0])
    return shape


def add_matrices(mat1, mat2):
    """that adds two matrices"""
    if mat_shape(mat1) != mat_shape(mat2):
        return None
    if type(mat1[0]) == int:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
