#!/usr/bin/env python3
"""performs the forward algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model"""
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])
    for i in range(1, T):
        X = np.dot(Transition.T, F[:, i - 1])
        alpha = Emission[:, Observation[i]]
        F[:, i] = np.multiply(alpha, X)
    P = F[:, T - 1].sum()
    return P, F
