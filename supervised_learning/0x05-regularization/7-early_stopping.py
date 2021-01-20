#!/usr/bin/env python3
"""determines if you should stop gradient descent early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """determines if you should stop gradient descent early"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return patience == count, count
