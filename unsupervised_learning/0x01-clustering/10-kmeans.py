#!/usr/bin/env python3
"""performs K-means on a dataset"""
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""
    return sklearn.cluster.k_means(X, k)[:2]
