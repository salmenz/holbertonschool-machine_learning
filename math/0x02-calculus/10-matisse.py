#!/usr/bin/env python3
"""calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if len(poly) == 1:
        if isinstance(poly[0], int):
            return [0]
        else:
            return None
    elif poly == []:
        return None
    else:
        poly = poly[1:]
        for i in range(len(poly)):
            if not isinstance(poly[i], int) and poly[i] < 0:
                return None
            else:
                poly[i] *= i+1
        return poly
