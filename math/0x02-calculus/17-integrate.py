#!/usr/bin/env python3
'''that calculates the integral of a polynomial'''


def poly_integral(poly, C=0):
    """that calculates the integral of a polynomial"""
    if type(C) != int and type(C) != float:
        return None
    if poly == []:
        return None
    for i in poly:
        if type(i) != int:
            return None
    l1 = []
    l1.append(C)
    for i in range(1, len(poly)+1):
        l1.append(poly[i-1]/i)
    for i in range(len(l1)):
        if l1[i] % 1 == 0:
            l1[i] = int(l1[i])
    return l1
