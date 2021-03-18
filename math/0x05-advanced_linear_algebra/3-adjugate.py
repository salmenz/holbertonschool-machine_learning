#!/usr/bin/env python3
"""calculates the adjugate of a matrix"""


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


def matrix_transpose(matrix):
    """"return transposed matrix"""
    m = []
    for i in range(len(matrix[0])):
        l1 = []
        for arr in matrix:
            l1.append(arr[i])
        m.append(l1)
    return m


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


def adjugate(matrix):
    """calculates the adjugate of a matrix"""
    if type(matrix) != list:
        raise TypeError("matrix must be a list of lists")
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if len(i) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
        if type(i) != list:
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        return [[1]]
    if len(matrix) >= 2:
        l1 = []
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                l1.append(determinant(drop(matrix, i, j)))
        for i in range(len(l1)):
            if i % 2:
                l1[i] *= -1
        mat = []
        for i in range(0, len(l1), len(matrix)):
            mat.append(l1[i:i + len(matrix)])
        if len(mat) == 2:
            mat[1][0] *= -1
            mat[1][1] *= -1
        return matrix_transpose(mat)
