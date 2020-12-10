#!/usr/bin/env python3
'''that calculates the integral of a polynomial'''


def poly_integral(poly, C=0):
    """that calculates the integral of a polynomial"""
    if type(poly) != list:
        return None
    for i in poly:
        if type(i) != int:
            return None
    if len(poly) == 1 and poly[0] == 0:
        return [C]
    l1 = []
    l1.append(C)
    for i in range(1, len(poly)+1):
        l1.append(poly[i-1]/i)
    for i in range(len(l1)):
        if l1[i] % 1 == 0:
            l1[i] = int(l1[i])
    return l1
