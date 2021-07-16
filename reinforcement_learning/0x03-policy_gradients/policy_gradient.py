#!/usr/bin/env python3
"""policy gradient"""
import numpy as np


def policy(matrix, weight):
    """compute to policy with a weight of a matrix"""
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / np.sum(exp)

def softmax_grad(softmax):
    """softmax"""
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def policy_gradient(state, weight):
    """compute the Monte-Carlo policy gradient"""
    action = np.argmax(policy(state, weight))
    softmax = softmax_grad(policy(state, weight))[action, :]
    log = softmax / policy(state, weight)[0, action]
    gradient = state.T.dot(log[None, :])
    return (action, gradient)
