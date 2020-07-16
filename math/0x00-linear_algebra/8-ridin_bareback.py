#!/usr/bin/env python3
'''Ridin’ Bareback'''


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
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
            for num1, num2 in zip(mat1[i], m[j]):
                p.append(num1 * num2)
            l1.append(sum(p))
        n.append(l1)
    return n
