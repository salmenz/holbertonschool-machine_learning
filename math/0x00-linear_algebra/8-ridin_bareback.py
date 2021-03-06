#!/usr/bin/env python3
'''Ridin’ Bareback'''


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    m = []
    for i in range(len(mat2[0])):
        l1 = []
        for arr in mat2:
            l1.append(arr[i])
        m.append(l1)
    n = []
    for i in range(len(mat1)):
        l1 = []
        for j in range(len(mat2[0])):
            p = []
            for k in range(len(mat1[i])):
                p.append(mat1[i][k] * m[j][k])
            l1.append(sum(p))
        n.append(l1)
    return n
