#!/usr/bin/env python3
"""class Poisson that represents a poisson distribution"""
e = 2.7182818285


class Poisson:
    """class Poisson"""
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
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """calcul pmf"""
        k = int(k)
        if k < 0:
            return 0
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        return e ** (-self.lambtha) * self.lambtha ** k / fact

    def cdf(self, k):
        """calcul cdf"""
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
