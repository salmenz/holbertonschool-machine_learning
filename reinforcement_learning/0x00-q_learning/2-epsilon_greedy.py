#!/usr/bin/env python3
"""Epsilon Greedy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Epsilon Greedy"""
    e = np.random.uniform(0, 1)
    if e < epsilon:
        acttion = np.random.randint(0, Q.shape[1])
    else:
        acttion = np.argmax(Q[state])
    return acttion
