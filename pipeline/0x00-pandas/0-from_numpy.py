#!/usr/bin/env python3
import pandas as pd
"""create a pd.DataFrame from a np.ndarray"""


def from_numpy(array):
    """create a pd.DataFrame from a np.ndarray"""
    alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    df = pd.DataFrame(data=array, columns=alpha[:array.shape[1]])
    return df
