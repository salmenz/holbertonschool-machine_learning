#!/usr/bin/env python3
"""performs pooling on images:"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images:"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_images = np.zeros((m, (h - kh) // sh + 1, (w - kw) // sw + 1, c))
    for i in range((h - kh) // sh + 1):
        for j in range((w - kw) // sw + 1):
            if mode == 'max':
                out_images[:, i, j, :] = np.max(images[
                    :, i*sh: i*sh + kh, j*sw: j*sw + kw, :], axis=(1, 2))
            else:
                out_images[:, i, j, :] = np.average(images[
                    :, i*sh: i*sh + kh, j*sw: j*sw + kw, :], axis=(1, 2))
    return out_images
