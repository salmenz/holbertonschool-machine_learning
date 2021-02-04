#!/usr/bin/env python3
"""performs back propagation over a convolutional layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """back propagation over a convolutional layer of a neural network"""
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros((A_prev.shape))
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c):
                    slice_A_prev = A_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, k]
                    slice_dA_prev = dA_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, k]
                    if mode == "max":
                        mask = (slice_A_prev == np.max(slice_A_prev))
                        slice_dA_prev += dA[i, h, w, k] * mask
                    else:
                        slice_dA_prev += dA[i, h, w, k]/(kh*kw)
    return dA_prev
