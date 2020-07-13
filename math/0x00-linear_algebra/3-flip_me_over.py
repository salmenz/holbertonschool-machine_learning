#!/usr/bin/env python3
def matrix_transpose(matrix):
    m = []
    for i in range(len(matrix[0])):
        l = []
        for arr in matrix:
            l.append(arr[i])
        m.append(l)
    return m