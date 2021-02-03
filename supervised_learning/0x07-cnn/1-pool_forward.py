#!/usr/bin/env python3
"""performs forward propagation over a pooling layer of a neural network:"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network"""
    m, h_prev, w_perv, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_images = np.zeros((m, (h_prev - kh) // sh + 1,
                          (w_perv - kw) // sw + 1, c_prev))
    for i in range((h_prev - kh) // sh + 1):
        for j in range((w_perv - kw) // sw + 1):
            if mode == 'max':
                out_images[:, i, j, :] = np.max(A_prev[
                    :, i*sh: i*sh + kh, j*sw: j*sw + kw, :], axis=(1, 2))
            else:
                out_images[:, i, j, :] = np.average(A_prev[
                    :, i*sh: i*sh + kh, j*sw: j*sw + kw, :], axis=(1, 2))
    return out_images
