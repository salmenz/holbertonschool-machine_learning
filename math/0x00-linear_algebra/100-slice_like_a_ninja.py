#!/usr/bin/env python3
'''Slice Like A Ninja'''


def np_slice(matrix, axes={}):
    """that slices a matrix along specific axes"""
    max = list(axes.keys())[-1] + 1
    m = []
    mat = []
    key = list(axes.keys())
    for i in range(max):
        if i not in key:
            m.append(slice(None, None, None))
        else:
            m.append(slice(*axes[i]))
    mat = matrix[tuple(m)]
    return (mat)
