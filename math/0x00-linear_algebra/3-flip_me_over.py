#!/usr/bin/env python3
'''Flip Me Over'''


def matrix_transpose(matrix):
    m = []
    for i in range(len(matrix[0])):
        l1 = []
        for arr in matrix:
            l1.append(arr[i])
        m.append(l1)
    return m
