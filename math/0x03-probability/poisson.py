#!/usr/bin/env python3
"""class Poisson that represents a poisson distribution"""


class Poisson:
    """class Poisson"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if not data:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data)/len(data)