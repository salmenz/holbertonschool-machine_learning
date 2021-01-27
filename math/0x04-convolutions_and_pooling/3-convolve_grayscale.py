#!/usr/bin/env python3
"""performs a convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if type(padding) is tuple:
        ph, pw = padding
        out_images = np.zeros((m, (h - kh + (2 * ph))//sh + 1,
                              (w - kw + (2 * pw))//sw + 1))
        images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    elif padding == 'valid':
        out_images = np.zeros((m, (h - kh)//sh+1, (w - kw)//sw+1))
        h -= kh + 1
        w -= kw + 1
    else:
        out_images = np.zeros((m, h//sh+1, w//sw+1))
        if kh % 2:
            h_pad = (kh - 1) // 2
        else:
            h_pad = kh // 2
        if kw % 2:
            w_pad = (kw - 1) // 2
        else:
            w_pad = kw // 2
        images = np.pad(images, ((0, 0), (h_pad, h_pad), (w_pad, w_pad)),
                        'constant')
    for i in range(0, h, sh):
        for j in range(0, w, sw):
            out_images[:, i//sh, j//sw] = np.sum(
                kernel * images[:, i: i + kh, j: j + kw], axis=(1, 2))
    return out_images
