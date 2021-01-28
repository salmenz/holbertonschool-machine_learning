#!/usr/bin/env python3
"""performs a convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    out_images = np.zeros((m, h - kh + (2 * ph) + 1, w - kw + (2 * pw) + 1))
    img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    for i in range(h - kh + 1):
        for j in range(w):
            out_images[:, i, j] = np.sum(
                kernel * img[:, i: i + kh, j: j + kw], axis=(1, 2))
    return out_images
