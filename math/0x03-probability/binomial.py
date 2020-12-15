#!/usr/bin/env python3
"""class Binomial that represents an normal distribution"""
e = 2.7182818285
pi = 3.1415926536


class Binomial:
    """class Binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        self.data = data
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            v = 0
            for i in data:
                v += (i - mean) ** 2
            v = v / len(data)
            p = 1 - (v / mean)
            self.n = round(mean / p)
            self.p = mean / self.n
