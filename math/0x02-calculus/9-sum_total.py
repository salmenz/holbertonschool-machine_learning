#!/usr/bin/env python3
'''summation_i_squared'''


def summation_i_squared(n):
    """summation_i_squared"""
    if not n or n <= 0:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)