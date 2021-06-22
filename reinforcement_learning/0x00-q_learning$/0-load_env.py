#!/usr/bin/env python3
"""Load the Environment"""
import numpy as np
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the Environment"""
    env = gym.make('FrozenLake-v0', map_name=map_name,
                   desc=desc, is_slippery=is_slippery)
    return env
