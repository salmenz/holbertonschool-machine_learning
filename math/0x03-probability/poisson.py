#!/usr/bin/env python3
"""class Poisson that represents a poisson distribution"""


class Poisson:
    """ Poisson distribution Class"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
