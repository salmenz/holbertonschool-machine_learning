#!/usr/bin/env python3
import numpy as np
def np_elementwise(mat1, mat2):
    a = np.array(mat1)
    b = np.array(mat2)
    return (np.add(a, b), np.subtract(a, b), np.multiply(a, b), np.true_divide(a, b))