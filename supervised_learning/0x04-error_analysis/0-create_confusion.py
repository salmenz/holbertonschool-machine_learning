#!/usr/bin/env python3
"""creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    ma = np.zeros((labels.shape[1], labels.shape[1]))
    for pred, valid in zip(labels, logits):
        ma[pred.argmax()][valid.argmax()] += 1
    return ma
