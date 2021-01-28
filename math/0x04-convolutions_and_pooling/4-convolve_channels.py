#!/usr/bin/env python3
"""performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on images with channels"""
    m, h, w = images.shape[:3]
    kh, kw = kernel.shape[:2]
    sh, sw = stride
    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    out_images = np.zeros((m, (h - kh + (2 * ph))//sh + 1,
                              (w - kw + (2 * pw))//sw + 1))
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    for i in range((h - kh + (2 * ph))//sh + 1):
        for j in range((w - kw + (2 * pw))//sw + 1):
            out_images[:, i, j] = np.sum(kernel * images[
                :, i*sh: i*sh + kh, j*sw: j*sw + kw, :], axis=(1, 2, 3))
    return out_images
