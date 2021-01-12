#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""
import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    v = [0]
    for i in range(len(data)):
        v.append(beta * v[i] + (1-beta) * data[i])
    d = []
    for i in range(len(v)-1):
        d.append(v[i+1]/(1 - beta ** (i+1)))
    return d
