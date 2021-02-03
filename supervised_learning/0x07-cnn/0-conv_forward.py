#!/usr/bin/env python3
"""performs forward propagation over a convolutional
layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional
    layer of a neural network"""
    m, h_prev, w_perv = A_prev.shape[:3]
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    ph, pw = 0, 0
    if padding == "same":
        ph = (((h_prev - 1) * sh + kh - h_prev) // 2) + (kh % 2 == 0)
        pw = (((w_perv - 1) * sw + kw - w_perv) // 2) + (kw % 2 == 0)
    out_images = np.zeros((m, (h_prev - kh + (2 * ph))//sh + 1,
                              (w_perv - kw + (2 * pw))//sw + 1, c_new))
    images = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    for i in range((h_prev - kh + (2 * ph))//sh + 1):
        for j in range((w_perv - kw + (2 * pw))//sw + 1):
            for n in range(c_new):
                out_images[:, i, j, n] = np.sum(W[:, :, :, n] * images[
                    :, i*sh: i*sh + kh, j*sw: j*sw + kw, :], axis=(1, 2, 3))
    return activation(out_images + b)
