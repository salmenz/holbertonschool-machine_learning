#!/usr/bin/env python3
def matrix_transpose(matrix):
    """"return trabsposed matrix"""
    m = []
    for i in range(len(matrix[0])):
        l1 = []
        for arr in matrix:
            l1.append(arr[i])
        m.append(l1)
    return m
