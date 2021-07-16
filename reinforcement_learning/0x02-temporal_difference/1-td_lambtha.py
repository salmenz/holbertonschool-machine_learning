#!/usr/bin/env python3
"""performs the TD(λ) algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """performs the TD(λ) algorithm"""
    episode = [[], []]
    x = [0 for i in range(env.observation_space.n)]
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            x = list(np.array(x) * lambtha * gamma)
            x[state] += 1
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            delta_t = reward + gamma * V[new_state] - V[state]
            V[state] = V[state] + alpha * delta_t * x[state]
            if done:
                break
            state = new_state
    return np.array(V)
