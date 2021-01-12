#!/usr/bin/env python3
"""that normalizes (standardizes) a matrix"""


def normalize(X, m, s):
    """that normalizes (standardizes) a matrix"""
    return (X - m) / s
