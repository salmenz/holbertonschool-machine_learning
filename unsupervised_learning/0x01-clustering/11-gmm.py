#!/usr/bin/env python3
"""calculates a GMM from a dataset"""
import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset"""
    g = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = g.weights_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)
    m = g.means_
    return pi, m, S, clss, bic
