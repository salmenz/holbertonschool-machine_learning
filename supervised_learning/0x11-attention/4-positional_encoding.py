#!/usr/bin/env python3
"""calculate the positional encoding for a transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """calculate the positional encoding for a transformer"""
    def calangels(position, i, d_model):
        rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * rates
    pos = calangels(np.arange(max_seq_len)[:, np.newaxis],
                           np.arange(dm)[np.newaxis, :], dm)
    pos[:, 0::2] = np.sin(pos[:, 0::2])
    pos[:, 1::2] = np.cos(pos[:, 1::2])
    return pos
