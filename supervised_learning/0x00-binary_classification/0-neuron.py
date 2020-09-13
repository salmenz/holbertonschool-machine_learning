#!/usr/bin/env python3
"""class neuron"""
import numpy as np


class Neuron:
    """class neuron"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.randn(nx)
        self.b = 0
        self.A = 0
