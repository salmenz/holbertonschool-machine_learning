#!/usr/bin/env python3
"""class Exponential that represents an exponential distribution"""
e = 2.7182818285


class Exponential:
    """class exponential"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data)/sum(data)

    def pdf(self, x):
        """calcul pdf"""
        if x < 0:
            return 0
        return self.lambtha * (e ** (- self.lambtha * x))

    def cdf(self, x):
        """calcul cdf"""
        if x < 0:
            return 0
        return 1 - e ** (- self.lambtha * x)
