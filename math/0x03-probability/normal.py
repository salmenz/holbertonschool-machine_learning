#!/usr/bin/env python3
"""class Normal that represents an normal distribution"""
e = 2.7182818285
pi = 3.1415926536


class Normal:
    """class normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data)/len(data)
            s = 0
            for i in data:
                s += (i - self.mean) ** 2
            self.stddev = (s/len(data)) ** 0.5

    def z_score(self, x):
        """calculate z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calculate x"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """calculate pdf"""
        v = (self.stddev * ((2 * pi) ** 0.5))
        return 1 / v * e ** (- 0.5 * ((x - self.mean) / self.stddev) ** 2)

    def erf(self, x):
        """calculate erf"""
        return (x - ((x ** 3) / 3) + ((x ** 5) / 10) - ((x ** 7) / 42) +
                                     ((x ** 9) / 216)) * (2 / pi ** (0.5))

    def cdf(self, x):
        """calculate cdf"""
        return 0.5 * (1 + self.erf((x - self.mean) /
                                   (self.stddev * (2 ** (0.5)))))
