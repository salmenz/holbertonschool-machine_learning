#!/usr/bin/env python3
"""calculates the minor of a matrix"""


def drop(matrix, i, j):
    """drop line and column"""
    l2 = []
    for x in range(len(matrix)):
        l1 = []
        if i != x:
            for y in range(len(matrix[x])):
                if y != j:
                    l1.append(matrix[x][y])
            l2.append(l1)
    return (l2)


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if len(matrix) > 2:
        sum = 0
        for j in range(len(matrix)):
            mat = drop(matrix, 0, j)
            sum += matrix[0][j] * ((-1) ** j) * determinant(mat)
        return sum


def minor(matrix):
    """calculates the minor of a matrix"""
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if type(i) != list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(i):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return [[1]]
    if len(matrix) >= 2:
        l1 = []
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                l1.append(determinant(drop(matrix, i, j)))
        mat = []
        for i in range(0, len(l1), len(matrix)):
            mat.append(l1[i:i + len(matrix)])
        return mat
