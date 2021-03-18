#!/usr/bin/env python3
"""calculates the determinant of a matrix"""


def drop(matrix, j):
    """drop a column"""
    l2 = []
    for x in range(len(matrix)):
        l1 = []
        for y in range(len(matrix[x])):
            if y != j:
                l1.append(matrix[x][y])
        l2.append(l1)
    return (l2)


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if matrix == [[]]:
        return 1
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if type(i) != list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(i):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if len(matrix) > 2:
        sum = 0
        for j in range(len(matrix)):
            mat = drop(matrix[1:], j)
            sum += matrix[0][j] * ((-1) ** j) * determinant(mat)
        return sum
