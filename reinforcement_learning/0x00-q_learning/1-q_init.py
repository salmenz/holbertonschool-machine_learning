#!/usr/bin/env python3
"""Initialize Q-table"""
import numpy as np
import gym


def q_init(env):
    """Initialize Q-table"""
    number_of_states = env.action_space.n
    number_of_actions = env.observation_space.n
    return np.zeros([number_of_actions, number_of_states])
