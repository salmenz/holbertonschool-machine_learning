#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2 regularization"""
    W_list = []
    for i in range(L):
        W_list.append(np.linalg.norm(weights['W' + str(i+1)]))
    return cost + lambtha/(m*2) * np.sum(W_list)
