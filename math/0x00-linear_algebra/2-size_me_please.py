#!/usr/bin/env python3
'''Size Me Please'''


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    a = []
    while type(matrix) is list:
        a.append(len(matrix))
        matrix = matrix[0]
    return a
