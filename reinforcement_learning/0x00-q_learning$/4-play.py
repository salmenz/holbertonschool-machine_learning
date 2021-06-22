#!/usr/bin/env python3
"""play"""
import numpy as np


def play(env, Q, max_steps=100):
    """trained agent play an episode"""
    rewards = []
    state = env.reset()
    done = False
    total_rewards = 0

    for step in range(max_steps):
        env.render()
        # Take the action based on the Q Table:
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        # If episode finishes:
        if done:
            env.render()
            rewards.append(total_rewards)
            break
        state = new_state
    return total_rewards
