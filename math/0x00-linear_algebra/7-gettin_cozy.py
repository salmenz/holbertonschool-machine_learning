#!/usr/bin/env python3
from copy import deepcopy
def cat_matrices2D(mat1, mat2, axis=0):
    m1 = []
    m2 = []
    m1 = deepcopy(mat1)
    m2 = deepcopy(mat2)
    if axis == 0:
        return m1+m2
    else:
        for i in range(len(m1)):
            for j in range(axis):
                m1[i].append(m2[i][j])
        return m1