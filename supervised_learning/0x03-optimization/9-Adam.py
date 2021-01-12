#!/usr/bin/env python3
"""updates a variable using the Adam optimization algorithm"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable using the Adam optimization algorithm"""
    v = beta1 * v + (1-beta1) * grad
    s = beta2 * s + (1-beta2) * (grad ** 2)
    vc = v/(1-(beta1 ** t))
    sc = s/(1-(beta2 ** t))
    var = var - alpha * (vc/((sc ** 0.5)+epsilon))
    return var, v, s
