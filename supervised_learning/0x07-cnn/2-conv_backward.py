#!/usr/bin/env python3
"""performs back propagation over a pooling layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """back propagation over a pooling layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    h_prev, w_prev, c_prev = A_prev.shape[1:]
    kh, kw = W.shape[:2]
    sh, sw = stride
    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = (((h_prev - 1) * sh + kh - h_prev) // 2) + 1
        pw = (((w_new - 1) * sw + kw - w_prev) // 2) + 1
    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    kernel = W[:, :, :, k]
                    dz = dZ[i, h, w, k]
                    slice_A = A_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, :]
                    dA[i, h*sh:h*sh+kh, w*sw:w*sw+kw, :] += dz * kernel
                    dW[:, :, :, k] += slice_A * dz
    dA = dA[:, ph:dA.shape[1]-ph, pw:dA.shape[2]-pw, :]
    return dA, dW, db
