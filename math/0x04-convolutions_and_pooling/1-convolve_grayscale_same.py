#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    out_images = np.zeros((m, h, w))
    if kh % 2:
        h_pad = (kh - 1) // 2
    else:
        h_pad = kh // 2
    if kw % 2:
        w_pad = (kw - 1) // 2
    else:
        w_pad = kw // 2
    img = np.pad(images, ((0, 0), (h_pad, h_pad), (w_pad, w_pad)), 'constant')
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            out_images[:, i, j] = np.sum(
                kernel * img[:, i: i + kh, j: j + kw], axis=(1, 2))
    return out_images
