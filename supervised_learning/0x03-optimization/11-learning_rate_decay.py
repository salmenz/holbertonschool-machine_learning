#!/usr/bin/env python3
"""updates the learning rate using inverse time decay in numpy"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse time decay in numpy"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
