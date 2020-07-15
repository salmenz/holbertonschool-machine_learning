#!/usr/bin/env python3
'''Line Up'''


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    m = []
    for i in range(len(arr1)):
        m.append(arr1[i] + arr2[i])
    return m
